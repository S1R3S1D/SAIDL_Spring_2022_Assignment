import torch
import numpy as np
import torch.nn as nn

#Model

class ConvNN(nn.Module):
  def __init__(self, emb_dim):

    super(ConvNN, self).__init__()

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

#Projection Head

class projection_head(nn.Module):
    def __init__(self, embed_dim=1024, output_dim=128):

        super(projection_head, self).__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, 2048),
            nn.ReLU(),

            nn.Linear(2048, self.output_dim)
        )

    def forward(self, x):
        x = self.projection(x)
        return x
