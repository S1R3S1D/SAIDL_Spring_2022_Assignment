import torch 
import torch.nn as nn
import gym
import torch.utils.data as data
import numpy as np
import pandas as pd
from matplotlib import image

#Training Models
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = torch.tensor(image_batch)
        image_batch = image_batch.float()/255.0
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

#Plotting Trained Models
def plot_random_env_states(encoder, decoder, data):
  sample = data[np.random.randint(0, 6000)]
  a = torch.tensor(np.expand_dims(sample, axis=0)).float()/255.0
  a = a.to(device)
  enc_data = encoder(a)
  dec_data = decoder(enc_data)
  dc = dec_data.cpu().squeeze().detach().numpy()
  plt.figure(figsize=(12, 10))
  f, axarr = plt.subplots(2, 2)
  axarr[0, 0].imshow(sample[0])
  axarr[0, 1].imshow(dc[0])
  axarr[1, 0].imshow(sample[1])
  axarr[1, 1].imshow(dc[1])
  plt.show()
