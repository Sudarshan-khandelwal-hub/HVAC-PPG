import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PPGAgent(nn.Module):    
    def __init__(self, obs_dim, action_dim, hidden_dim=64, action_low = None, action_high = None):
        super(PPGAgent, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Shared network
        self.shared_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(self.device)
        
        # Policy network
        self.policy_mean = nn.Linear(hidden_dim, action_dim).to(self.device)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim)).to(self.device)

        self.value = nn.Linear(hidden_dim, 1).to(self.device)
        self.aux_value = nn.Linear(hidden_dim, 1).to(self.device)


        #Action range
        self.action_low = torch.tensor(action_low).to(self.device) if action_low is not None else torch.zeros(action_dim).to(self.device)
        self.action_high = torch.tensor(action_high).to(self.device) if action_high is not None else torch.ones(action_dim).to(self.device)

        # Value network
        self.value = nn.Linear(hidden_dim, 1).to(self.device)
        
        # Auxiliary value network
        self.aux_value = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, obs):
        return self.shared_network(obs.to(self.device))

    def get_policy(self, obs):
        shared_features = self.forward(obs)
        mean = self.policy_mean(shared_features)
        mean = torch.tanh(mean)
        std = self.policy_log_std.exp().clamp(min=1e-6,max=1)
        return Normal(mean, std)

    def get_value(self, obs):
        return self.value(self.forward(obs)).squeeze(-1)

    def get_aux_value(self, obs):
        shared_features = self.forward(obs)
        return self.aux_value(shared_features)

    def get_action_and_value(self, obs, action=None):
        shared_features = self.forward(obs)
        mean = self.policy_mean(shared_features)
        mean = torch.tanh(mean)
        log_std = self.policy_log_std.clamp(min=-20, max=2)
        std = log_std.exp()
        policy = Normal(mean, std)
        
        if action is None:
            action = policy.sample()
        
        scaled_action = self.action_low + (self.action_high - self.action_low) * (action + 1) / 2
        
        return (scaled_action, 
                policy.log_prob(action).sum(-1), 
                policy.entropy().sum(-1), 
                self.value(shared_features).squeeze(-1))
    
    def get_pi_value_and_aux_value(self, obs):
        shared_features = self.forward(obs)
        
        policy = self.get_policy(obs)
        value = self.value(shared_features)
        aux_value = self.aux_value(shared_features)
        
        return policy, value, aux_value

class PPGActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, action_low=None, action_high=None):
        super(PPGActorCritic, self).__init__()
        
        self.actor = PPGAgent(obs_dim, action_dim, hidden_dim, action_low, action_high)
        self.critic = PPGAgent(obs_dim, action_dim, hidden_dim, action_low, action_high)

    def get_action_and_value(self, obs, action=None):
        return self.actor.get_action_and_value(obs, action)

    def get_value(self, obs):
        return self.critic.get_value(obs)

    def get_pi_value_and_aux_value(self, obs):
        return self.actor.get_pi_value_and_aux_value(obs)

    def get_pi(self, obs):
        return self.actor.get_policy(obs)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class PPGBuffer:
    def __init__(self, obs_dim, action_dim, buffer_size, num_envs, device):
        self.device = device
        self.obs = torch.zeros((buffer_size, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((buffer_size, num_envs), device=device)
        self.dones = torch.zeros((buffer_size, num_envs), device=device)
        self.values = torch.zeros((buffer_size, num_envs), device=device)
        self.returns = torch.zeros((buffer_size, num_envs), device=device)
        self.advantages = torch.zeros((buffer_size, num_envs), device=device)
        self.logprobs = torch.zeros((buffer_size, num_envs), device=device)
        self.ptr, self.size = 0, 0
        self.buffer_size = buffer_size
        self.num_envs = num_envs

    def add(self, obs, action, reward, done, value, logprob):
        self.obs[self.ptr] = obs.to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.rewards[self.ptr] = reward.to(self.device)
        self.dones[self.ptr] = done.to(self.device)
        self.values[self.ptr] = value.to(self.device)
        self.logprobs[self.ptr] = logprob.to(self.device)
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        last_value = last_value.to(self.device)
        last_gae_lam = torch.zeros(self.num_envs, device=self.device)
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self):
        return (self.obs[:self.size], self.actions[:self.size], 
                self.rewards[:self.size], self.dones[:self.size], 
                self.values[:self.size], self.returns[:self.size], 
                self.advantages[:self.size], self.logprobs[:self.size])

    def clear(self):
        self.ptr, self.size = 0, 0
