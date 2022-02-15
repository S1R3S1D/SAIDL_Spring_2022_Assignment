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

#The Environment
env = gym.make('PongNoFrameskip-v4')
env = gym.wrappers.AtariPreprocessing(env, 30, 4, 84, False, True)
env = gym.wrappers.FrameStack(env, 4)
env.unwrapped.get_action_meanings()

#Hyperparameters, loss function, optimizer and others

lr = 0.001

dqn = DQN(env.action_space.n)

target_network = copy.deepcopy(dqn)
target_network.load_state_dict(dqn.state_dict())

sync_freq = 5000

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(dqn.parameters(), lr = lr)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

dqn.to(device)
target_network.to(device)

#Training DQN with a Target Network and Experience Replay

def train_dqn_w_trgnet_expreplay(n_eps):

  replay = deque(maxlen=30000)
  mini_batch_size = 32
  gamma = 0.99
  episode = 0
  losses = []
  
  for _ in range(n_eps):

    print("Training Episode :", episode)
    epsilon = 1
    episode+=1
    epoch = 0
    env.reset()

    for _ in tqdm(range(100000)):
      (state1, reward1, done1, info1) = env.step(np.random.randint(0, env.action_space.n))
      
      state1_ = torch.tensor(np.expand_dims(np.array(state1), axis=0)).float()/255.0
      state1_ = torch.Tensor(state1_)
      state1_ = state1_.to(device)

      qval_ = dqn(state1_)
      qval = qval_.cpu().detach().squeeze().numpy()

      if random.random()<epsilon:
        action = np.random.randint(0, env.action_space.n)
      else:
        action = np.argmax(qval)

      if epsilon>0.1:
        epsilon-=1/100000
      (state2, reward2, done2, info2) = env.step(action)

      if done2:
        env.reset()

      state2_ = torch.from_numpy(np.expand_dims(np.array(state2), axis=0)).float()/255.0
      state2_ = torch.Tensor(state2_)
      state2_ = state2_.to(device)

      exp = (state1_, action, state2_, reward2, done2)
      replay.append(exp)

      state1 = state2

      if len(replay)>5000:

        mini_batch = random.sample(replay, mini_batch_size)

        state1_batch = torch.cat([s1 for (s1, a, s2, r, d) in mini_batch]).to(device)
        state2_batch = torch.cat([s2 for (s1, a, s2, r, d) in mini_batch]).to(device)
        action_batch = torch.Tensor([a for (s1, a, s2, r, d) in mini_batch]).to(device)
        reward_batch = torch.Tensor([r for (s1, a, s2, r, d) in mini_batch]).to(device)
        done_batch = torch.Tensor([d for (s1, a, s2, r, d) in mini_batch]).to(device)

        Q1 = dqn(state1_batch)
        with torch.no_grad():
          Q2 = target_network(state2_batch)

        Y = reward_batch + gamma * ((1-done_batch)*torch.max(Q2, dim=1)[0])
        X = Q1.gather(dim = 1, index = action_batch.long().unsqueeze(dim=1)).squeeze()

        loss = loss_fn(X, Y)

        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        epoch+=1
        if epoch%sync_freq ==0:
          target_network.load_state_dict(dqn.state_dict())
    if episode%2==0:
      test_dqn(env)

  losses = np.array(losses)
  return losses
