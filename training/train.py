import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from model import IQN
from env import TrackmaniaEnv
import os
import tqdm


class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = IQN().to(self.device)
        self.target_net = IQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.90
        self.target_update = 10
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return (
                random.random() > 0.5,  # gas
                random.random() > 0.5,  # brake
                random.uniform(-100, 100),  # steer
            )

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            gas, brake, steer = self.policy_net(state)
            return (gas.item() > 0.5, brake.item() > 0.5, steer.item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states and next_states to tensors - using stack instead of array
        states = torch.FloatTensor(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(
            self.device
        )
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Split actions into separate components
        gas_actions = torch.FloatTensor([float(a[0]) for a in actions]).to(self.device)
        brake_actions = torch.FloatTensor([float(a[1]) for a in actions]).to(
            self.device
        )
        steer_actions = torch.FloatTensor([float(a[2]) for a in actions]).to(
            self.device
        )

        current_gas, current_brake, current_steer = self.policy_net(states)
        next_gas, next_brake, next_steer = self.target_net(next_states)

        # Calculate TD targets for each output
        gas_target = rewards + (1 - dones) * self.gamma * next_gas
        brake_target = rewards + (1 - dones) * self.gamma * next_brake
        steer_target = rewards + (1 - dones) * self.gamma * next_steer

        # Calculate losses using the actual actions taken
        gas_loss = nn.MSELoss()(current_gas, gas_actions)
        brake_loss = nn.MSELoss()(current_brake, brake_actions)
        steer_loss = nn.MSELoss()(current_steer, steer_actions)

        total_loss = gas_loss + brake_loss + steer_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train():
    env = TrackmaniaEnv("AI: 1")
    agent = DQNAgent()
    episodes = 1000

    for episode in tqdm.trange(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

        # Update target network
        if episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(
            f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}"
        )

        # Save model every 100 episodes
        if episode % 50 == 0:
            torch.save(
                agent.policy_net.state_dict(),
                f"models/v2/model_checkpoint_{episode}.pth",
            )


if __name__ == "__main__":
    train()
