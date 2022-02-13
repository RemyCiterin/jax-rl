
import numpy as np
import os

from collections import deque
import gym
import cv2
cv2.ocl.setUseOpenCL(False)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)

        self.deque = deque(maxlen=k)
        self.k = k

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.deque.append(obs)

        return np.concatenate(self.deque, axis=-1), reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k): self.deque.append(obs)

        return np.concatenate(self.deque, axis=-1)


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, x=84, y=84, gray=True):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(0, 255, shape=(x, y, 1 if gray else 3), dtype=np.uint8)
        self.gray = gray
        self.x = x
        self.y = y

    def observation(self, obs):
        if self.gray:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            return cv2.resize(
                obs, (self.x, self.y), interpolation=cv2.INTER_AREA
            )[..., None]

        else:
            return cv2.resize(
                obs, (self.x, self.y), interpolation=cv2.INTER_AREA
            )


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def make_env(ident, render=False, gray=True, stack=True):
    if render: env = gym.make(ident, render_mode="human")
    else: env = gym.make(ident)

    env = MaxAndSkipEnv(ResizeWrapper(env, gray=gray))

    if stack: return FrameStack(env)
    return env