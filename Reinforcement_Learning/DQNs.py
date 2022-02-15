import gym
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from collections import deque
from tqdm import tqdm
import random
import copy
import matplotlib.pyplot as plt

#Deep Q Network Acrhitecture

class DQN(nn.Module):
  def __init__(self, n_actions):
    
    super(DQN, self).__init__()

    self.conv_net = nn.Sequential(
        
        nn.Conv2d(4, 32, 8, 4),
        nn.ReLU(True),

        nn.Conv2d(32, 64, 4, 2),
        # nn.BatchNorm2d(12),
        nn.ReLU(True),

        nn.Conv2d(64, 64,3, 1),
        # nn.MaxPool2d(3, 2),
        nn.ReLU(True)
    )

    self.flatten = nn.Flatten(start_dim = 1)

    self.fc_layer = nn.Sequential(
        nn.Linear(64*7*7, 512),
        nn.ReLU(True),
        
        nn.Linear(512, n_actions),
    )

  def forward(self, x):

    x = self.conv_net(x)
    x = self.flatten(x)
    x = self.fc_layer(x)

    return x
  
