import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import wandb
import config 
from utils.helpers import scale_action, safe_mean
from agents.ppg_agent import PPGBuffer  # Import PPGBuffer

def ppg_train(env, agent, total_timesteps, num_steps, n_pi, e_aux, learning_rate=3e-4, clip_coef=0.2, 
              ent_coef=0.01, vf_coef=0.5, beta_clone=1.0, gamma=0.99, gae_lambda=0.95, 
              device="cuda" if torch.cuda.is_available() else "cpu", results_dir=None, args=None):
    
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    
    global_step = 0
    
    # Initialize PPGBuffer
    buffer = PPGBuffer(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        buffer_size=num_steps,
        num_envs=env.num_envs,
        device=device
    )

    # Initialize environment
    next_obs, _ = env.reset()
    next_obs = torch.from_numpy(np.array(next_obs)).float().to(device)
    next_done = torch.zeros(env.num_envs).to(device)
    num_updates = total_timesteps // (num_steps * env.num_envs)
    
    episode_reward = 0
    comfort_score = 0
    energy_consumption = 0

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so
        if isinstance(agent, nn.Module):
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Policy phase
        for step in range(0, num_steps):
            global_step += 1 * env.num_envs

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.to(device))

            raw_action = action.cpu().numpy()
            scaled_action = scale_action(raw_action)

            next_obs, reward, done, truncated, info = env.step(scaled_action)

            episode_reward += reward[0]
            energy_consumption += next_obs[13]

            next_obs = torch.from_numpy(np.array(next_obs)).float().to(device)
            reward = torch.tensor(reward[0]).float().to(device)  # Convert reward to tensor
            done = torch.tensor(done).float().to(device)

            # Store experience in buffer
            buffer.add(next_obs, action, reward, done, value.flatten(), logprob)  # Flatten value tensor

            next_done = done

            if done.any() or truncated:
                print(f"Episode finished. Reward: {episode_reward}, Comfort: {comfort_score}, Energy: {energy_consumption}")
                if args and args.track:
                    wandb.log({
                        "episode_reward": episode_reward,
                        "comfort_score": comfort_score,
                        "HVAC_electricity_demand_rate": energy_consumption
                    })
                next_obs, _ = env.reset()
                next_obs = torch.from_numpy(np.array(next_obs)).float().to(device)

        # Compute returns and advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            buffer.compute_returns_and_advantages(next_value, gamma, gae_lambda)

        # Get data from buffer
        b_obs, b_actions, b_rewards, b_dones, b_values, b_returns, b_advantages, b_logprobs = buffer.get()
        
        # Flatten the batch
        b_obs = b_obs.reshape((-1,) + env.observation_space.shape)
        b_actions = b_actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = b_logprobs.reshape(-1)
        b_advantages = b_advantages.reshape(-1)
        b_returns = b_returns.reshape(-1)
        b_values = b_values.reshape(-1)
        
        # Optimizing the policy and value network
        b_inds = np.arange(num_steps * env.num_envs)
        for _ in range(n_pi):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps * env.num_envs, env.num_envs):
                end = start + env.num_envs
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2)
                pg_loss = safe_mean(pg_loss)
                
                # Value loss
                v_loss = F.mse_loss(newvalue.view(-1), b_returns[mb_inds])
                
                entropy_loss = safe_mean(entropy)
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        # Auxiliary phase
        for _ in range(e_aux):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps * env.num_envs, env.num_envs):
                end = start + env.num_envs
                mb_inds = b_inds[start:end]
                
                old_pi, old_value, _ = agent.get_pi_value_and_aux_value(b_obs[mb_inds])
                new_pi, new_value, new_aux_value = agent.get_pi_value_and_aux_value(b_obs[mb_inds])
                
                # KL divergence for policy distillation
                kl_loss = kl_divergence(old_pi, new_pi).mean()
                
                # Auxiliary value loss
                aux_v_loss = 0.5 * ((new_aux_value - b_returns[mb_inds]) ** 2).mean()
                
                # Joint loss
                joint_loss = aux_v_loss + beta_clone * kl_loss
                
                optimizer.zero_grad()
                joint_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        # Clear the buffer after each update
        buffer.clear()
    
    return agent