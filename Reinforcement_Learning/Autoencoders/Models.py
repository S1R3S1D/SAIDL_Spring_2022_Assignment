import torch 
import torch.nn as nn

#Convolutional Encoder

class Encoder(nn.Module):
  def __init__(self, latent_space_dim):
    
    super(Encoder, self).__init__()

    self.encoder_conv = nn.Sequential(
        
        nn.Conv2d(4, 16, 5, 2, 2),
        nn.ReLU(True),

        nn.Conv2d(16, 32, 3, 2, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(True),

        nn.Conv2d(32, 64, 5, 2, 0),
        nn.ReLU(True)
    )

    self.flatten = nn.Flatten(start_dim=1)

    self.encoder_lin = nn.Sequential(
        nn.Linear(9*9*64, 512),
        nn.ReLU(True),

        nn.Linear(512, latent_space_dim)
    )

  def forward(self, x):
    
    x = self.encoder_conv(x)
    x = self.flatten(x)
    x = self.encoder_lin(x)

    return x
  
#Convolutional Decoder

class Decoder(nn.Module):
  def __init__(self, latent_space_dim):
    
    super(Decoder, self).__init__()

    self.decoder_lin = nn.Sequential(
        nn.Linear(latent_space_dim, 512),
        nn.ReLU(True),

        nn.Linear(512, 9*9*64),
        nn.ReLU(True)
    )

    self.unflatten = nn.Unflatten(dim = 1, unflattened_size=(64, 9, 9))

    self.decoder_conv = nn.Sequential(
        nn.ConvTranspose2d(64, 32, 5, 2, padding=0, output_padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(True),

        nn.ConvTranspose2d(32, 16, 3, 2, 1, 0),
        nn.BatchNorm2d(16),
        nn.ReLU(True),

        nn.ConvTranspose2d(16, 4, 5, 2, 1, 1)
    )

  def forward(self, x):
    x = self.decoder_lin(x)
    x = self.unflatten(x)
    x = self.decoder_conv(x)
    x = torch.sigmoid(x)
    return x

