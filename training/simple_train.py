from env import GymTrackmaniaEnv
from make_instances import make_n_instances
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(rank):
    def _init():
        env = GymTrackmaniaEnv(rank - 1, f"AI: {rank}")
        return env

    return _init


if __name__ == "__main__":
    multi = True
    num_of_instances = 4

    if multi:
        make_n_instances(num_of_instances)
        input("Press Enter to continue...")

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

    policy_kwargs = dict(net_arch=[512, 512], dueling=True)

    model = DQN(
        "CnnPolicy",
        env,
        batch_size=64,
        buffer_size=100_000,
        learning_rate=linear_schedule(1e-5, 5e-6),
        exploration_final_eps=0.02,
        exploration_fraction=0.1,
        train_freq=(4, "step"),
        target_update_interval=10_000,
        verbose=2,
        tensorboard_log="./logs",
        seed=42,
        tau=0.001,
        policy_kwargs=policy_kwargs,
        optimize_memory_usage=True,
    )

    adjusted_save_freq = max(50_000 // num_of_instances, 1)

    checkpoint_callback = CheckpointCallback(
        save_freq=adjusted_save_freq,
        save_path="./models/v12/",
        name_prefix="dqn_trackmania",
    )

    model.learn(total_timesteps=300_000, callback=checkpoint_callback)
