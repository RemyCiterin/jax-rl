
try:
    from faster_fifo import Queue
    import faster_fifo_reduction
except:
    import multiprocessing as mp
    class Queue:
        def __init__(self, *args, **kargs):
            self.queue = mp.Queue()

        def put(self, *args, **kargs):
            return self.queue.put(*args, **kargs)

        def get(self, *args, **kargs):
            return self.queue.get(*args, **kargs)

from queue import Full, Empty

import multiprocessing as mp
import time

import numpy as np

from utils.SharedArray import SharedArray

TIMEOUT = 120


class AsyncEnv(object):

    def run_env(ident, env_fn, shared_obs, Qin, Qout, m, render=False):

        env_list = []
        for _ in range(m):
            env_list.append(env_fn())

        for i in range(m):
            shared_obs[i].re_init()
            shared_obs[i].set(env_list[i].reset())
            Qout.put(ident)

        while True:
            action, i = Qin.get(timeout=TIMEOUT)


            if i is None:
                [env.close() for env in env_list]
                return

            obs, reward, done, _ = env_list[i].step(action)
            if done: obs = env_list[i].reset()

            if render and i == 0: env_list[i].render()

            shared_obs[i].set(obs)
            Qout.put((reward, done, ident, i))


    def __init__(self, env_fun, shape, dtype, n, m=1, batch_size=1, render=False, sleep_at_init=False):
        self.Qout = [Queue() for _ in range(n)]
        self.Qin  = Queue()

        self.shared_obs = [[SharedArray(shape, dtype) for _ in range(m)] for _ in range(n)]

        ctx = mp.get_context('fork')

        for i in range(n):
            process = ctx.Process(
                target=AsyncEnv.run_env, args=(i, env_fun, self.shared_obs[i], self.Qout[i], self.Qin, m, render and i == 0), daemon=True)
            if sleep_at_init: time.sleep(0.1)
            process.start()

        for _ in range(n*m): self.Qin.get(timeout=TIMEOUT)

        for i in range(n):
            for j in range(m):
                self.Qin.put((0, False, i, j))

        self.batch_size = batch_size

        self.shape = shape
        self.dtype = dtype
        
        self.n = n
        self.m = m

    def send(self, actions, ident):
        for i in range(self.batch_size):
            self.Qout[ident[i] // self.m].put((actions[i], ident[i] % self.m))
    
    def recv(self):
        obs = np.zeros((self.batch_size, *self.shape), dtype=self.dtype)
        reward = np.zeros((self.batch_size,), dtype=float)
        ident = np.zeros((self.batch_size,), dtype=int)
        done = np.zeros((self.batch_size,), dtype=bool)
        
        for k in range(self.batch_size):
            r, d, i, j = self.Qin.get(timeout=TIMEOUT)

            self.shared_obs[i][j].copyto(obs[k])
            ident[k] = i * self.m + j
            reward[k] = r
            done[k] = d
        
        return obs, reward, done, ident

    def close(self):
        for i in range(self.n):
            self.Qout[i].put((None, None))

    def re_init(self):
        for i in range(self.n):
            for j in range(self.m):
                self.shared_obs[i][j].re_init()


import gym
import gym.spaces
import numpy as np

class SeqVecEnv(gym.Env):
    def __init__(self, env_fn, n):
        self.env_list = [env_fn() for _ in range(n)]

        obs_space = self.env_list[0].observation_space
        action_space = self.env_list[0].action_space
        self.n = n

        self.observation_space = gym.spaces.Box(
            low=np.array([obs_space.low for _ in range(n)]), 
            high=np.array([obs_space.high for _ in range(n)]), 
            shape=(n,)+obs_space.shape,
            dtype=obs_space.dtype
        )

        self.action_space = action_space

    def reset(self):
        return np.stack([env.reset() for env in self.env_list], axis=0)

    def step(self, actions):
        obs, reward, done = [], [], []

        for env, action in zip(self.env_list, actions):
            o, r, d, _ = env.step(action)
            if d: o = env.reset()
            reward.append(r)
            done.append(d)
            obs.append(o)

        return np.stack(obs, axis=0), np.array(reward, dtype=float), np.array(done, dtype=bool), {}

    def render(self, *args, **kargs):
        self.env_list[0].render(*args, **kargs)

    def close(self):
        [env.close() for env in self.env_list]

class SyncEnv(object):
    def run_env(ident, env_fn, shared_obs, Qin, Qout, m, batch_size, render=False):

        env_list = []
        for _ in range(m):
            env_list.append(SeqVecEnv(env_fn, batch_size))

        for i in range(m):
            shared_obs[i].re_init()
            shared_obs[i].set(env_list[i].reset())
            Qout.put(ident)

        while True:
            action, i = Qin.get(timeout=TIMEOUT)

            if i is None: 
                [env.close() for env in env_list]
                return

            obs, reward, done, _ = env_list[i].step(action)

            if i == 0 and render: env_list[i].render()

            shared_obs[i].set(obs)
            Qout.put((reward, done, ident, i))

    def __init__(self, env_fun, shape, dtype, n, m=1, batch_size=1, render=False, sleep_at_init=False):
        self.Qout = [Queue() for _ in range(n)]
        self.Qin  = Queue()

        self.shared_obs = [[SharedArray((batch_size,) + shape, dtype) for _ in range(m)] for _ in range(n)]

        ctx = mp.get_context('fork')

        for i in range(n):
            process = ctx.Process(target=SyncEnv.run_env, args=(
                i, env_fun, self.shared_obs[i], self.Qout[i], self.Qin, m, batch_size, render and i == 0), daemon=True)
            if sleep_at_init: time.sleep(0.1)
            process.start()

        for _ in range(n*m): self.Qin.get(timeout=TIMEOUT)

        for i in range(n):
            for j in range(m):
                self.Qin.put((np.zeros((batch_size,)), np.zeros((batch_size,)), i, j))

        self.batch_size = batch_size

        self.shape = shape
        self.dtype = dtype
        
        self.n = n
        self.m = m

    def send(self, actions, ident):
        self.Qout[ident // self.m].put((actions, ident % self.m))
    
    def recv(self):
        reward, done, i, j = self.Qin.get(timeout=TIMEOUT)
        obs = self.shared_obs[i][j].get()
        ident = i * self.m + j

        return obs, reward, done, ident

    def close(self):
        for i in range(self.n):
            self.Qout[i].put((None, None))

    def re_init(self):
        for i in range(self.n):
            for j in range(self.m):
                self.shared_obs[i][j].re_init()


