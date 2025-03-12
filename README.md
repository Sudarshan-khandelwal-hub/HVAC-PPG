# Reinforcement Learning for HVAC Control in Data Centers

This repository contains implementations of advanced reinforcement learning algorithms for optimizing HVAC (Heating, Ventilation, and Air Conditioning) control in data center environments. The project uses the Sinergym framework to simulate data center thermal dynamics and energy consumption.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running PPG Training](#running-ppg-training)
  - [Hyperparameter Optimization with GA](#hyperparameter-optimization-with-ga)
- [Environment Setup](#environment-setup)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

Data centers represent approximately 1-2% of global electricity consumption, with a significant portion dedicated to cooling systems. This project aims to develop smart HVAC control strategies using reinforcement learning to reduce energy consumption while maintaining appropriate temperature and humidity levels for optimal equipment operation.

The implementation uses the [Sinergym](https://github.com/ugr-sail/sinergym) framework, which provides realistic building simulation environments for reinforcement learning research, coupled with EnergyPlus for accurate physics-based modeling.

## ✨ Features

- Implementation of Phasic Policy Gradient (PPG) algorithm for HVAC control
- Genetic Algorithm (GA) for hyperparameter optimization
- Multi-objective reward functions balancing energy efficiency and comfort
- Support for various weather conditions and building configurations
- Comprehensive logging and visualization tools
- Integration with Weights & Biases for experiment tracking

## 🧠 Algorithms

### Phasic Policy Gradient (PPG)

PPG is an advanced actor-critic reinforcement learning algorithm that separates policy and value function learning phases. This implementation features:

- Shared network architecture with policy and value heads
- Auxiliary value function for improved value estimation
- Policy distillation mechanism for stable learning
- Generalized Advantage Estimation (GAE)

### Genetic Algorithm (GA)

The GA implementation is used for hyperparameter optimization, including:

- Population-based evolution to find optimal hyperparameters
- Customizable genetic operations (selection, crossover, mutation)
- Metrics for diversity and convergence tracking
- Visualization tools for monitoring the optimization process

## 📂 Project Structure

```
.
├── agents/
│   ├── __init__.py
│   └── ppg_agent.py           # PPG agent implementation
├── algorithms/
│   ├── ga.py                  # Genetic Algorithm implementation
│   └── ppg.py                 # PPG training algorithm
├── utils/
│   ├── __init__.py
│   └── helpers.py             # Utility functions and helper classes
├── core_ppg.py                # Main script for PPG training
├── ga_ppg.py                  # Script for GA-based hyperparameter optimization
└── README.md                  # This file
```

## 🛠️ Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install torch numpy gymnasium wandb
pip install sinergym
```

3. Clone this repository:

```bash
git clone https://github.com/yourusername/rl-hvac-datacenter.git
cd rl-hvac-datacenter
```

## 🚀 Usage

### Running PPG Training

To train a PPG agent on the data center environment:

```bash
python core_ppg.py --env Eplus-datacenter-hot-continuous-v1 --total_timesteps 100000
```

Optional arguments:
- `--num_steps`: Number of steps per policy rollout (default: 2048)
- `--n_pi`: Number of policy updates per iteration (default: 32)
- `--e_aux`: Number of auxiliary epochs (default: 6)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--device`: Device to run on (default: "cuda" if available, else "cpu")
- `--track`: Enable tracking with wandb (default: False)

### Hyperparameter Optimization with GA

To optimize PPG hyperparameters using the Genetic Algorithm:

```bash
python ga_ppg.py --env Eplus-datacenter-hot-continuous-v1 --total_timesteps 50000
```

The GA will search for optimal hyperparameters and train a final model using the best configuration found.

## 🌐 Environment Setup

The project uses custom Sinergym environments with specific variables, actuators, and meters defined for data center simulation. The key components include:

- **Observation Space**: Includes temperature, humidity, outdoor conditions, and energy consumption metrics
- **Action Space**: Controls heating and cooling setpoints
- **Reward Function**: Multi-objective function balancing energy consumption and thermal comfort
- **Weather Data**: Uses TMY3 weather files for realistic outdoor conditions

Configuration is handled in both `core_ppg.py` and `ga_ppg.py` through the `make_env` function, which sets up the environment with appropriate wrappers.
