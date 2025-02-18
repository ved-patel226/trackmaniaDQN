from tminterface.client import Client
from tminterface.interface import TMInterface
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space, flatten
from gymnasium import spaces
import numpy as np
import os
import cv2
from track_viewer import find_nearest_checkpoint

# from tminterface2 import TMInterface
import mss
import random
import pyautogui

try:
    from .screenshot import capture_window
    from .make_instances import make_n_instances
except ImportError:
    from screenshot import capture_window
    from make_instances import make_n_instances
import time
from tqdm import tqdm
import msvcrt


class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.completed = False

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")
        iface.set_speed(1)

    def on_checkpoint_count_changed(
        self, iface: TMInterface, current: int, target: int
    ) -> None:
        if current == target:
            self.completed = True
            iface.prevent_simulation_finish()
            self.restart_run(iface)

    def map_steering(
        self, num: int, in_min=-100, in_max=100, out_min=-65536, out_max=65536
    ) -> int:
        return int((num - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    def move(self, iface: TMInterface, brake: bool, steer: bool):
        iface.set_input_state(
            accelerate=True if not brake else False,
            brake=True if brake else False,
            left=True if steer is False else False,
            right=True if steer is True else False,
        )

    def register(self, iface: TMInterface):
        iface.register(self)

    def restart_run(self, iface: TMInterface):
        iface.give_up()
        time.sleep(1)
        iface.set_input_state(
            accelerate=1,
            steer=0,
        )
        time.sleep(1.5)

    def observe(self, iface: TMInterface, window_title: str, sct: mss.mss):
        self.completed = False
        iface._process_server_message()

        state = iface.get_simulation_state()

        if state.position[2] > 590:
            self.completed = True

        return (
            {
                "speed": state.display_speed,
                "position": state.position,
                "velocity": state.velocity,
                "yaw_pitch_roll": state.yaw_pitch_roll,
                "picture": capture_window(window_title, sct),
            },
            True if self.completed else False,
        )

    def get_time(self, iface: TMInterface):
        return iface.get_simulation_state().time

    def crossed_finish_line(
        self, iface: TMInterface, finish_line_x: float = 100.0
    ) -> bool:
        state = iface.get_simulation_state()
        # Check if the car's x position has passed the finish line threshold.
        return state.position[0] >= finish_line_x

    def get_internal_data(self, iface: TMInterface, target, attr) -> list:
        try:
            data = getattr(iface.get_simulation_state(), target)
            data = str(data)
            occurrences = []
            index = data.find(attr)
            while index != -1:
                end_index = data.find("\n", index)
                data_str = data[index:end_index]
                occurrences.append(data_str.split(":")[1].strip() == "True")
                index = data.find(attr, end_index)
            return occurrences
        except Exception:
            return []


class TrackmaniaEnv:
    def __init__(self, server_number: int, window_title: str | None = None) -> None:
        self.client = MainClient()
        self.iface = TMInterface(f"TMInterface{server_number}")
        print("Created client and interface")

        self.client.register(self.iface)
        print("Registered client")

        with tqdm(total=1, desc="Waiting for registration", leave=False) as pbar:
            while not self.iface.registered:
                time.sleep(0.1)
                print(server_number, window_title)
            pbar.update(1)

        self.sct = mss.mss()

        self.window_title = window_title
        self.time_stuck = 0

        self.server_number = server_number
        self.zone_centers = np.load("map2.npy")

    def get_rewards(self, obs_dict: dict, completed: bool, brake: bool) -> float:
        # Track previous speed; if not set, initialize it to current speed.
        if not hasattr(self, "last_speed"):
            self.last_speed = obs_dict["speed"]

        if not hasattr(self, "last_checkpoint"):
            self.last_checkpoint = 0

        current_speed = obs_dict["speed"]

        # Check for sudden speed drop (20 or more)
        sudden_drop_penalty = 0
        if self.last_speed - current_speed >= 20:
            sudden_drop_penalty = -50  # Apply an additional penalty

        # Update last_speed for the next call
        self.last_speed = current_speed

        (distance, checkpoint_idx) = find_nearest_checkpoint(
            obs_dict["position"], self.zone_centers
        )

        if completed:
            return 100  # reduced from 1000
        elif obs_dict["position"][1] < 140:
            return -100  # reduced from -1000
        elif current_speed < 20:
            # Scale reward from low speed: now within [-100, -5]
            reward = np.interp(current_speed, [0, 20], [-100, -5])
        elif brake:
            reward = -current_speed
        else:
            # Scale down the reward for high speed
            reward = (current_speed - 50) ** 1.1 / 10.0

        reward += sudden_drop_penalty

        if checkpoint_idx >= self.last_checkpoint:
            self.last_checkpoint = checkpoint_idx
            bonus = np.interp(distance, [0, 30], [100, 0])
            reward += bonus

        return reward

    def step(
        self, action: tuple[bool, bool] | None
    ) -> tuple[dict, float, bool, bool, dict]:
        if action:
            self.client.move(self.iface, *action)

        obs_dict, completed = self.client.observe(
            self.iface, self.window_title, self.sct
        )

        truncated = (
            self.client.get_time(self.iface) > 25000 * 2
            or obs_dict["position"][1] < 100
        )

        if obs_dict["speed"] < 5:
            self.time_stuck += 1
        else:
            self.time_stuck = 0

        if self.time_stuck >= 10:
            truncated = True

        reward = self.get_rewards(obs_dict, completed, False)

        return obs_dict, reward, completed, truncated, {}

    def reset(self) -> dict:
        # time.sleep(5)
        # print("Resetting environment")

        # if self.window_title:
        #     windows = pyautogui.getWindowsWithTitle(self.window_title)
        #     if windows:
        #         try:
        #             window = windows[0]
        #             try:
        #                 window.activate()
        #                 window.restore()
        #             except Exception as e:
        #                 if "Error code from Windows: 0" not in str(e):
        #                     raise e
        #             x, y = window.center
        #             pyautogui.click(x, y)

        #             time.sleep(0.01)
        #             y += 220
        #             for i in range(10):
        #                 pyautogui.click(x, y - i)
        #                 pyautogui.click(x, y + i)

        #         except Exception as e:
        #             print(f"Error clicking window: {e}")
        #     else:
        #         print(f"No window found with title: {self.window_title}")

        self.last_checkpoint = 0
        self.last_speed = 0
        self.client.restart_run(self.iface)
        time.sleep(1)
        return self.client.observe(self.iface, self.window_title, self.sct)[0]

    @property
    def action_space(self):
        class _ActionSpace:
            def sample(inner_self):
                return (False, self.client.map_steering(random.randint(-100, 100)))

        return _ActionSpace()


class GymTrackmaniaEnv(gym.Env):
    def __init__(self, server_number: int, window_title: str | None = None):
        super(GymTrackmaniaEnv, self).__init__()
        self.env = TrackmaniaEnv(server_number, window_title)

        # Define action space
        self.action_space = spaces.Discrete(5)
        # Original Dict space for observations
        self.dict_space = spaces.Dict(
            {
                "speed": spaces.Box(low=0.0, high=1000.0, shape=(), dtype=np.float32),
                "position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "velocity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "yaw_pitch_roll": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "picture": spaces.Box(
                    low=0, high=255, shape=(128, 128, 1), dtype=np.uint8
                ),
            }
        )
        # Flatten the Dict into a Box for the observation space
        self.observation_space = flatten_space(self.dict_space)

    def step(self, action):
        conversion_chart = {
            0: (False, False),
            1: (False, True),
            2: (True, False),
            3: (True, True),
            4: (False, 0),
        }

        brake, steer = conversion_chart[action]

        # Take action in the environment
        obs_dict, reward, done, truncated, info = self.env.step((brake, steer))

        # Flatten the observation dict to a numpy array
        flattened_obs = flatten(self.dict_space, obs_dict)
        return flattened_obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs_dict = self.env.reset()
        flattened_obs = flatten(self.dict_space, obs_dict)
        return flattened_obs, {}

    def close(self):
        # Properly cleanup resources if necessary.
        self.env.sct.close()


def main() -> None:
    # make_n_instances(1)
    # input("Press Enter to continue...")

    env = GymTrackmaniaEnv(0, "AI: 1")
    # env.reset()

    while True:

        print(env.env.step(None)[1])
        time.sleep(0.1)


if __name__ == "__main__":
    main()
