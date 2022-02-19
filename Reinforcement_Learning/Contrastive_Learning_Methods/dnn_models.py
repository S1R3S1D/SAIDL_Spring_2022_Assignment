import torch
import numpy as np
import torch.nn as nn

class CNN(nn.Module):
  def __init__(self, emb_dim):
    
    super(CNN, self).__init__()

    self.conv_net = nn.Sequential(
        
        nn.Conv2d(4, 32, 8, 4),
        nn.ReLU(),

        nn.Conv2d(32, 64, 4, 2),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 64,3, 1),
        nn.ReLU()
    )

    self.flatten = nn.Flatten(start_dim = 1)

    self.fc_layer = nn.Sequential(
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        
        nn.Linear(512, emb_dim),
    )

  def forward(self, x):

    x = self.conv_net(x)
    x = self.flatten(x)
    x = self.fc_layer(x)

    return x
