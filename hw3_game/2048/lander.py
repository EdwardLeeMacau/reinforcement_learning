import argparse
import gymnasium as gym

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

def make_env(mode=None):
    env = gym.make('LunarLander-v2', render_mode=mode)
    return Monitor(env, filename=None, allow_early_resets=True)

# Set hyper params (configurations) for training
env = DummyVecEnv([make_env])
config = {
    # Experiments
    "run_id": "lander",
    "save_path": "models/lander",

    # Hyperparameters
    "algorithm": PPO,
    "policy_network": "MlpPolicy",

    "learning_rate": 0.0003,                    # Learning rate
    "n_epochs": 20,                             # Number of steps to optimize
    "n_steps": 512,                             # Number of sample steps per update
    "clip_range": 0.2,                          # PPO clip range
    "batch_size": 64,                           # Mini-batch size

    "gamma": 0.99,                              # MDP discount factor
    "log_interval": 1000,                      # Log interval
    "timesteps_per_epoch": 204800,
    "eval_episode_num": 10,
}

def train():
    # TODO: Parallelize training envs
    train_env = DummyVecEnv([make_env for _ in range(8)])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = PPO(
        policy=config["policy_network"],
        env=train_env,
        learning_rate=config["learning_rate"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        clip_range=config["clip_range"],
        batch_size=config["batch_size"],
        verbose=1,
        tensorboard_log=f'runs/{config["run_id"]}'
    )

    model.learn(
        total_timesteps=config["timesteps_per_epoch"],
        reset_num_timesteps=False,
        callback=EvalCallback(
            eval_env=env,
            eval_freq=config["log_interval"],
            n_eval_episodes=config["eval_episode_num"],
            deterministic=True,
            # render=True,
            # eval_episodic_rewards=True,
            verbose=1,
            best_model_save_path=config["save_path"],
        ),
    )

def render():
    def make_env():
        env = gym.make('LunarLander-v2', render_mode='human')
        return Monitor(env, filename=None, allow_early_resets=True)

    env = DummyVecEnv([make_env])

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = PPO.load("models/lander/best_model.zip", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render("human")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--render", action="store_true", help="Render the agent")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.train:
        train()

    if args.render:
        render()

if __name__ == "__main__":
    main()
