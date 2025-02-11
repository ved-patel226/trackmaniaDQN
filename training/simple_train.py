from env import GymTrackmaniaEnv
from make_instances import make_n_instances
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack


def make_env(rank):
    def _init():
        env = GymTrackmaniaEnv(rank - 1, f"AI: {rank}")
        return env

    return _init


if __name__ == "__main__":
    multi = True
    num_of_instances = 4

    if multi:
        # make_n_instances(num_of_instances)
        # input("Press Enter to continue...")

        env_fns = [make_env(i) for i in range(1, num_of_instances + 1)]
        env = SubprocVecEnv(env_fns)
    else:
        make_n_instances(1)
        input("Press Enter to continue...")
        env = GymTrackmaniaEnv(0, "AI: 1")

    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    os.system("cls")

    def linear_schedule(initial_value, final_value):
        def func(progress_remaining):
            return final_value + (initial_value - final_value) * progress_remaining

        return func

    # model = PPO("MlpPolicy", env, verbose=2)
    model = DQN(
        "MlpPolicy",
        env,
        batch_size=128,
        buffer_size=40_000,
        learning_rate=linear_schedule(1e-4, 5e-5),
        exploration_final_eps=0.02,
        exploration_fraction=0.2,
        train_freq=(4, "step"),
        target_update_interval=500,
        verbose=2,
        tensorboard_log="./logs",
        seed=42,
    )

    i = 1
    while True:
        model.learn(total_timesteps=200_000, reset_num_timesteps=False)
        model.save(f"models/v11/dqn_trackmania_{i * 200_000}")
        i += 1
