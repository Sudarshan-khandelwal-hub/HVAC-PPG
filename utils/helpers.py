import numpy as np
import torch
import gym
import sinergym
import math
from sinergym.utils.wrappers import *

def setup_environment(env_name, seed=None, config_params=None):
    """
    Set up the Sinergym environment with appropriate wrappers.
    
    Args:
        env_name (str): Name of the Sinergym environment.
        seed (int, optional): Random seed for reproducibility.
        config_params (dict, optional): Configuration parameters for the environment.
    
    Returns:
        gym.Env: The configured Sinergym environment.
    """
    env = sinergym.make(env_name, config_params=config_params)
    env = NormalizeObservation(env)
    env = LoggerWrapper(env)
    if seed is not None:
        env.seed(seed)
    return env

def compute_returns(rewards, last_value, gamma):
    """
    Compute the returns (discounted sum of rewards).
    
    Args:
        rewards (list): List of rewards for each timestep.
        last_value (float): The estimated value of the final state.
        gamma (float): Discount factor.
    
    Returns:
        list: The computed returns for each timestep.
    """
    returns = []
    R = last_value
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    return returns

def compute_advantages(rewards, values, gamma, lambda_):
    """
    Compute the Generalized Advantage Estimation (GAE).
    
    Args:
        rewards (list): List of rewards for each timestep.
        values (list): List of estimated values for each state.
        gamma (float): Discount factor.
        lambda_ (float): GAE parameter.
    
    Returns:
        list: The computed advantages for each timestep.
    """
    advantages = []
    last_advantage = 0
    last_value = values[-1]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * last_value - values[t]
        advantage = delta + gamma * lambda_ * last_advantage
        advantages.insert(0, advantage)
        last_advantage = advantage
        last_value = values[t]
    return advantages

def normalize(x):
    """
    Normalize a vector.
    
    Args:
        x (numpy.ndarray): Input vector.
    
    Returns:
        numpy.ndarray: Normalized vector.
    """
    return (x - x.mean()) / (x.std() + 1e-8)

def explained_variance(y_pred, y_true):
    """
    Computes the explained variance regression score.
    
    Args:
        y_pred (numpy.ndarray): Predicted values.
        y_true (numpy.ndarray): True values.
    
    Returns:
        float: Explained variance score.
    """
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_obs_action_dims(env):
    """
    Get the dimensions of observation and action spaces.
    
    Args:
        env (gym.Env): The environment.
    
    Returns:
        tuple: (observation dimension, action dimension)
    """
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]
    return obs_dim, action_dim

def create_log_gaussian(mean, log_std, t):
    """
    Create a log-gaussian distribution.
    
    Args:
        mean (torch.Tensor): Mean of the distribution.
        log_std (torch.Tensor): Log standard deviation of the distribution.
        t (torch.Tensor): Input tensor.
    
    Returns:
        torch.Tensor: Log-probability of the distribution.
    """
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    """
    Numerically stable logsumexp.
    
    Args:
        inputs (torch.Tensor): Input tensor.
        dim (int, optional): Dimension along which the logsumexp is computed.
        keepdim (bool): Whether to keep the dimension.
    
    Returns:
        torch.Tensor: Result of the logsumexp operation.
    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    """
    Perform a soft update of the target network parameters.
    
    Args:
        target (nn.Module): Target network.
        source (nn.Module): Source network.
        tau (float): Interpolation parameter.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    """
    Perform a hard update of the target network parameters.
    
    Args:
        target (nn.Module): Target network.
        source (nn.Module): Source network.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def scale_action(action):
        # Clip actions to [-1, 1] range
        noisy_action = np.clip(action, -1, 1)
        
        # Scale and clip heating setpoint
        heating_action = float(15 + (noisy_action[0] + 1) * (22.00 - 15) / 2)
        heating_action = np.clip(heating_action, 15, 22.00)
        
        # Scale and clip cooling setpoint
        cooling_action = float(22.00 + (noisy_action[1] + 1) * (30 - 22.00) / 2)
        cooling_action = np.clip(cooling_action, 22.00, 30)
        
        return [heating_action, cooling_action]


class CustomCSVLogger(CSVLogger):
    def __init__(
            self,
            monitor_header: str,
            progress_header: str,
            log_progress_file: str,
            log_file: Optional[str] = None,
            flag: bool = True):
        super(CustomCSVLogger, self).__init__(monitor_header,progress_header,log_progress_file,log_file,flag)
        self.last_10_steps_reward = [0]*10

    def _create_row_content(
            self,
            obs: List[Any],
            action: Union[int, np.ndarray, List[Any]],
            terminated: bool,
            truncated: bool,
            info: Optional[Dict[str, Any]]) -> List:

        if info.get('reward') is not None:
            self.last_10_steps_reward.pop(0)
            self.last_10_steps_reward.append(info['reward'])

        return [
            info.get('timestep',0)] + list(obs) + list(action) + [
            info.get('time_elapsed(hours)',0),
            info.get('reward',None),
            np.mean(self.last_10_steps_reward),
            info.get('total_power_no_units'),
            info.get('comfort_penalty'),
            info.get('abs_comfort'),
            terminated,
            truncated]
    
        
def safe_mean(tensor):
    return tensor.mean() if not torch.isnan(tensor).any() else torch.tensor(0.0).to(tensor.device)

        
import logging
import os
from datetime import datetime

class GALogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('GA_Logger')
        self.logger.setLevel(logging.INFO)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'ga_log_{timestamp}.txt')
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_generation(self, generation, best_fitness, avg_fitness, best_individual):
        self.logger.info(f"Generation {generation}:")
        self.logger.info(f"  Best Fitness: {best_fitness}")
        self.logger.info(f"  Average Fitness: {avg_fitness}")
        self.logger.info(f"  Best Individual: {best_individual.genes}")
        self.logger.info("------------------------")

    def log_final_results(self, best_individual, total_generations, total_time):
        self.logger.info("Final Results:")
        self.logger.info(f"  Best Individual: {best_individual.genes}")
        self.logger.info(f"  Best Fitness: {best_individual.fitness}")
        self.logger.info(f"  Total Generations: {total_generations}")
        self.logger.info(f"  Total Time: {total_time}")

import matplotlib.pyplot as plt
import seaborn as sns

class GAVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_fitness_progress(self, generations, best_fitnesses, avg_fitnesses):
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitnesses, label='Best Fitness')
        plt.plot(generations, avg_fitnesses, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Progress Over Generations')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'fitness_progress.png'))
        plt.close()

    def plot_parameter_distribution(self, all_individuals, generation):
        params = all_individuals[0][1].genes.keys()
        fig, axes = plt.subplots(len(params), 1, figsize=(10, 5*len(params)))
        for i, param in enumerate(params):
            values = [ind.genes[param] for _, ind in all_individuals if ind[0] == generation]
            sns.histplot(values, ax=axes[i], kde=True)
            axes[i].set_title(f'{param} Distribution - Generation {generation}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'param_distribution_gen_{generation}.png'))
        plt.close()

import numpy as np

class GAMetrics:
    @staticmethod
    def diversity(population):
        genes = np.array([list(ind.genes.values()) for ind in population])
        return np.mean(np.std(genes, axis=0))

    @staticmethod
    def improvement_rate(fitness_history, window=5):
        if len(fitness_history) < window:
            return 0
        recent = fitness_history[-window:]
        return (recent[-1] - recent[0]) / window

    @staticmethod
    def convergence(fitness_history, threshold=1e-6, window=10):
        if len(fitness_history) < window:
            return False
        recent = fitness_history[-window:]
        return np.std(recent) < threshold
    

