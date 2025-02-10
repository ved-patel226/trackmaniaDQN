import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from env import TrackmaniaEnv
import cv2
from tqdm import trange
import glob


class IQN(nn.Module):
    def __init__(self):
        super(IQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        # Separate output heads for gas, brake, and steer
        self.fc_gas = nn.Linear(512, 1)
        self.fc_brake = nn.Linear(512, 1)
        self.fc_steer = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 7 * 7 * 64)
        x = torch.relu(self.fc1(x))
        # Gas and brake outputs as probabilities (True if >0.5)
        gas = torch.sigmoid(self.fc_gas(x))
        brake = torch.sigmoid(self.fc_brake(x))
        # Steer output scaled to [-100, 100]
        steer = torch.tanh(self.fc_steer(x)) * 100
        return gas, brake, steer
