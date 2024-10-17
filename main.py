import argparse
import json
import os
from datetime import datetime
import time

import numpy as np
import torch
import wandb
import gymnasium as gym

import sinergym
from sinergym.utils.wrappers import NormalizeObservation, LoggerWrapper , ReduceObservationWrapper , MultiObjectiveReward,  NormalizeAction

from agents.ppg_agent import PPGAgent
from algorithms.ppg import ppg_train
from utils.helpers import scale_action , CustomCSVLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Eplus-datacenter-hot-continuous-v1", help="Sinergym environment name")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps of the experiments")
    parser.add_argument("--num_steps", type=int, default=2048, help="Number of steps per policy rollout")
    parser.add_argument("--n_pi", type=int, default=32, help="Number of policy updates per iteration")
    parser.add_argument("--e_aux", type=int, default=6, help="Number of auxiliary epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="Surrogate clipping coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.2, help="Value function coefficient")
    parser.add_argument("--beta_clone", type=float, default=1.0, help="Behavior cloning coefficient")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--track", type=bool, default=False, help="Track with wandb")
    return parser.parse_args()


new_time_variables=['month', 'day_of_month', 'hour']

new_variables={
            'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
            'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
            'wind_speed': ('Site Wind Speed', 'Environment'),
            'wind_direction': ('Site Wind Direction', 'Environment'),
            'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
            'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
            'west_zone_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'West Zone'),
            'east_zone_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'East Zone'),
            'west_zone_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'West Zone'),
            'east_zone_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'East Zone'),
            'west_zone_air_temperature': ('Zone Air Temperature', 'West Zone'),
            'east_zone_air_temperature': ('Zone Air Temperature', 'East Zone'),
            'west_zone_air_humidity': ('Zone Air Relative Humidity', 'West Zone'),
            'east_zone_air_humidity': ('Zone Air Relative Humidity', 'East Zone'),
            'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building')
            }

new_meters={
            'east_zone_electricity':'Electricity:Zone:EAST ZONE',
            'west_zone_electricity':'Electricity:Zone:WEST ZONE',
            }

new_actuators = {
            'Heating_Setpoint_RL': (
                'Schedule:Compact',
                'Schedule Value',
                'Heating Setpoints'),
            'Cooling_Setpoint_RL': (
                'Schedule:Compact',
            'Schedule Value',
            'Cooling Setpoints')
}


def make_env(env_id, seed):
    def thunk():
        global new_variables
        global new_actuators
        global new_time_variables
        env = gym.make(env_id,
                       variables= new_variables,
                       actuators=new_actuators,
                       meters=new_meters,
                       time_variables=new_time_variables,
                       weather_files = ['USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw','USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw','USA_CO_Aurora-Buckley.Field.ANGB.724695_TMY3.epw'],
                       weather_variability=(1.0,0.0,0.001))
        env = MultiObjectiveReward(env,reward_terms=['energy_term','comfort_term'])
        env = NormalizeObservation(env)
        env = ReduceObservationWrapper(env, obs_reduction=['wind_speed','wind_direction'])
        env = LoggerWrapper(env,logger_class=CustomCSVLogger,monitor_header = ['timestep'] + env.get_wrapper_attr('observation_variables') +
                env.get_wrapper_attr('action_variables') + ['time (hours)', 'reward', '10-mean-reward',
                'power_penalty', 'comfort_penalty', 'terminated', 'truncated'])

        print(f"Observation space: {env.observation_space}")
        print(f"Observation space shape: {env.observation_space.shape}")
        print(f"Action space: {env.action_space}")
        
        if seed is not None:
            env.reset(seed=seed)
        return env
    return thunk



def test_agent(env, agent, num_episodes=30):
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _, _, _ = agent.get_action_and_value(torch.FloatTensor(obs).to(args.device))
            scaled_action = scale_action(action.cpu().numpy())
            obs, reward, done, truncated, info = env.step(scaled_action)
            episode_reward += reward[0]
            if done or truncated:
                break
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} reward: {episode_reward}")
    return np.mean(total_rewards)

def main(args):
    # Set up wandb
    if args.track:
        run_name = f"PPG_{args.env}_{args.seed}_{int(time.time())}"
        wandb.init(
            project="hvac_ppg_datacenter",
            entity="your_entity",
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create directories for saving results
    run_name = f"PPG_{args.env}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    results_dir = os.path.join("experiments", "experiment_results", "results", run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create environment
    env = make_env(args.env, args.seed)()
    
    # Initialize agent
    agent = PPGAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    ).to(args.device)

    # Train agent
    trained_agent = ppg_train(
        env=env,
        agent=agent,
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        n_pi=args.n_pi,
        e_aux=args.e_aux,
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        beta_clone=args.beta_clone,
        device=args.device,
        results_dir=results_dir,
        args=args
    )
    
    # Save trained model
    model_path = os.path.join(results_dir, "trained_agent.pth")
    try:
        torch.save(trained_agent.state_dict(), model_path)
        print(f"Model saved successfully at: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        # Attempt to save in the current directory as a fallback
        fallback_path = "trained_agent.pth"
        try:
            torch.save(trained_agent.state_dict(), fallback_path)
            print(f"Model saved in current directory: {fallback_path}")
        except Exception as e:
            print(f"Failed to save model in current directory: {e}")

    # Test trained agent
    test_reward = test_agent(env, trained_agent)
    print(f"Average test reward: {test_reward}")

    if args.track:
        wandb.log({"test_reward": test_reward})
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)