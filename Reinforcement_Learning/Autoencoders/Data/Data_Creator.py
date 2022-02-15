import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.data as data
import os

#Dataset wrapper for atari gym environments

!rm -rf Data

class OpenAIGymData(Dataset):
  def __init__(self, env, fire = False):
    
    os.mkdir('Data')
    os.mkdir('Data/AtariEnvImageData')
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, screen_size=84, terminal_on_life_loss=True, grayscale_obs=True)
    env = gym.wrappers.FrameStack(env, 4)
    env.reset()
    if fire:
      env.step(1)
    data_folder = 'Data/AtariEnvImageData'
    self.data_folder = data_folder
    for i in range(12):
      file_name = data_folder+'/'+str(i)+'.npy'
      with open(file_name, 'wb') as f:
        samples = []
        for i in range(500):
          action = np.random.randint(0, env.action_space.n)
          (sample, reward, done, info) = env.step(action)
          if done == True:
            env.reset()
            if fire:
              env.step(1)

          sample = np.array(sample)
          sample = sample.astype(float)/255.0
          samples.append(sample)
        samples = np.array(samples)
        np.save(f, samples)

  def __len__(self):
    return 500*12

  def __getitem__(self, index):
    
    file_no = str(int(index/500))
    in_file_index = index%500

    file_name = self.data_folder+'/'+file_no+'.npy'
    with open(file_name, 'rb') as f:
      sample = np.load(f)
      return sample[in_file_index]
