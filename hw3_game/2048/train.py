from datetime import datetime
from typing import Dict

import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from torch import nn, Tensor
from torch.nn import functional as F

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

def unstack(layered, layers=16):
    """Convert a [layers, 4, 4] representation into [4, 4] with one layers for each value."""
    # representation is what each layer represents
    representation = (2 ** (torch.arange(layers, dtype=int, device=layered.device) + 1))

    # layered is the flat board repeated layers times
    flat = torch.permute(layered, (0, 2, 3, 1))

    # Now set the values in the board to 1 or zero depending whether they match representation.
    # Representation is broadcast across a number of axes
    flat = torch.sum(flat * representation, axis=-1)
    return flat

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # Hardcoded feature dim for other parts of the code
        # super().__init__(observation_space, features_dim=8448 + 48 + 8)
        super().__init__(observation_space, features_dim=64 + 256 + 48 + 8)

        # 3x3 conv, 2x2 conv
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=2)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # Utilities
        n = x.shape[0]

        # * Board occupancy (Raw features)
        occupancy = torch.sum(x, dim=1, keepdim=True)                       # dim: (n, 1, 4, 4), rng: {0, 1}
        flat = torch.log2(unstack(x))
        flat[flat == -torch.inf] = 0                                        # dim: (n, 4, 4)   , rng: {0, 1, ..., 16}

        # * Current max tile and pos (Raw features)
        tile, _ = torch.max(flat.flatten(start_dim=1), dim=1)
        tile = tile.to(torch.int64)                                         # dim: (n, )       , rng: {0, 1, ..., 15}

        f1 = F.one_hot(tile, num_classes=16).float()                        # dim: (n, 16)     , rng: {0, 1}
        f2 = torch.sum(x * f1.view(n, 16, 1, 1), dim=1, keepdim=True)       # dim: (n, 1, 4, 4), rng: {0, 1}

        # * Check movability (Raw features)
        # check if different in flat[:, :, i] and flat[:, :, i+1]
        row = (~torch.logical_and(
            torch.all(flat[:, :, :-1] != flat[:, :, 1:], dim=2),
            torch.all(flat, dim=2)
        )).to(dtype=torch.float32)                                          # dim: (n, 4)      , rng: {0, 1}
        col = (~torch.logical_and(
            torch.all(flat[:, :-1, :] != flat[:, 1:, :], dim=1),
            torch.all(flat, dim=1)
        )).to(dtype=torch.float32)                                          # dim: (n, 4)      , rng: {0, 1}

        # * Aggregated features
        f = torch.cat(
            tuple(map(
                lambda t: t.flatten(start_dim=1),
                [occupancy, f1, f2, row, col]
            )), dim=1
        )                                                                   # dim: (n, 48 + 8)     , rng: {0, 1}

        # * Learnable features
        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)

        return torch.cat((x2.flatten(start_dim=1), x.flatten(start_dim=1), f), dim=1)

class PolicyNetwork(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, use_expln=True,
            features_extractor_class=FeatureExtractor,
            net_arch={'pi': [256, 128, 64], 'vf': [256, 128, 64]}
        )

# Piecewise linear schedule for PPO clip parameter
def clip_range(current_progress_remaining: float) -> float:
    return 0.15 * current_progress_remaining + 0.05

# Set hyper params (configurations) for training
now = datetime.now().strftime("%Y%m%d-%H%M%S")
my_config = {
    # Experiments
    "run_id": "PPOv27",
    "save_path": "models/PPOv27",

    # Hyperparameters
    "algorithm": PPO,
    "policy_network": PolicyNetwork,

    "learning_rate": 0.0003,                    # Learning rate
    "n_epochs": 20,                             # Number of steps to optimize
    "n_steps": 512,                             # Number of sample steps per update
    "clip_range": 0.2,                          # PPO clip range
    "batch_size": 64,                           # Mini-batch size

    "gamma": 0.99,                              # MDP discount factor
    "epoch_num": 1000,
    "log_interval": 10000,                      # Log interval
    "timesteps_per_epoch": 20480,
    "eval_episode_num": 10,
}

def make_env():
    env = gym.make('2048-v0')
    return Monitor(env, filename=None, allow_early_resets=True)

def train(env, model: BaseAlgorithm, config: Dict):

    current_best = 0

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=EvalCallback(
            #     eval_env=env,
            #     eval_freq=config["log_interval"],
            #     n_eval_episodes=config["eval_episode_num"],
            #     deterministic=True,
            #     render=True,
            #     # eval_episodic_rewards=True,
            #     verbose=1,
            #     # best_model_save_path=config["save_path"],
            # ),
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score = 0
        avg_highest = 0
        for seed in range(config["eval_episode_num"]):
            done = False

            # Set seed using old Gym API
            env.seed(seed)
            obs = env.reset()

            # Interact with env using old Gym API
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

            avg_highest += info[0]['highest']/config["eval_episode_num"]
            avg_score   += info[0]['score']/config["eval_episode_num"]

        print("Avg_score:  ", avg_score)
        print("Avg_highest:", avg_highest)
        print()

        ### Save best model
        if current_best < avg_score:
            print("Saving Model")
            current_best = avg_score
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")

        print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    # TODO: Parallelize training envs
    train_env = DummyVecEnv([make_env for _ in range(8)])
    env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = PPO(
        policy=my_config["policy_network"],
        env=train_env,
        learning_rate=my_config["learning_rate"],
        n_epochs=my_config["n_epochs"],
        gamma=my_config["gamma"],
        n_steps=my_config["n_steps"],
        clip_range=my_config["clip_range"],
        batch_size=my_config["batch_size"],
        verbose=1,
        tensorboard_log=f'runs/{my_config["run_id"]}'
    )

    # TODO: Env for CPU training
    # env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)
    # model = A2C(
    #     policy=my_config["policy_network"],
    #     env=env,
    #     learning_rate=my_config["learning_rate"],
    #     gamma=my_config["gamma"],
    #     n_steps=my_config["n_steps"],
    #     # clip_range=my_config["clip_range"],
    #     # batch_size=my_config["batch_size"],
    #     verbose=1,
    #     # device='cpu',
    #     tensorboard_log=f'runs/{my_config["run_id"]}'
    # )
    train(env, model, my_config)
