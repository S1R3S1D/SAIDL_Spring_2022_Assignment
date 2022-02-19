import torch
import numoy as np
import gym
import PIL
from PIL import Image
from collections import Deque

#Custom gym wrappers on atari environments 

#This wrapper returns a gray scaled 84X84 image as a numpy array(as original) from an original atari environment
class cropped84x84_grayscale(gym.ObservationWrapper):
    def __init__(self, env):
        super(cropped84x84_grayscale, self).__init__(env)

    def observation(self, observation):
        observation = Image.fromarray(observation)

        observation = observation.crop((0, 35, 160, 190))

        observation = observation.resize((84, 84), Image.LANCZOS)

        observation = observation.convert('L')

        observation = np.asarray(observation)

        return observation

#Returns the element wise maximum over the last 2 consecutive frames and skips frames 
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip = 4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward+=reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
