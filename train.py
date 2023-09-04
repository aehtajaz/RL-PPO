import numpy as np

from Agent import TimeLimitWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import GrayScaleObservation

import os
import retro

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Process the rewards
                modified_rewards = []

                for reward_str in y:
                    if isinstance(reward_str, int):
                        reward_float = float(reward_str)
                        modified_rewards.append(reward_float)
                    elif "." in str(reward_str):
                        reward_str = str(reward_str)  # Ensure reward_str is a string
                        first_dot_index = reward_str.find(".")
                        part1 = reward_str[:first_dot_index + 1]  # Include the first dot
                        part2 = reward_str[first_dot_index + 1:].replace(".", "")
                        reward_str = part1 + part2

                        try:
                            reward_float = float(reward_str)
                            modified_rewards.append(reward_float)
                        except ValueError:
                            print(f"Warning: Failed to convert reward value to float: {reward_str}")
                    else:
                        try:
                            reward_float = float(reward_str)
                            modified_rewards.append(reward_float)
                        except ValueError:
                            print(f"Warning: Failed to convert reward value to float: {reward_str}")

                
                # Convert the modified rewards to a suitable data type
                y = np.array(modified_rewards, dtype=np.float32)

                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving the best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True



# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        #env = gym.make(env_id)
        env = retro.make(game=env_id)
        env = TimeLimitWrapper(env, max_steps=2000)
        env = MaxAndSkipEnv(env, 4)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_id = "SuperMarioBros-Nes"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]),"tmp/TestMonitor")

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    #env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=0.00003)
    #model = PPO.load("tmp/best_model", env=env)
    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=500000, callback=callback, tb_log_name="PPO-00003")
    model.save(env_id)
    print("------------- Done Learning -------------")
    env = retro.make(game=env_id)
    env = TimeLimitWrapper(env)