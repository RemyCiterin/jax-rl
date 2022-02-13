import os 
os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "false"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

from envs.wrapper import make_env
def env_fn():
    return make_env("BreakoutNoFrameskip-v4", gray=True, stack=True)

from utils.conventions import *

from model.Q_model import *
from algorithms.Q_IMPALA import *

from jax.random import PRNGKey
import jax.numpy as jnp

def make_agent():
    agent = Agent(CONV_MODEL, (84, 84 ,4), 4, use_H_target=True)
    agent.obs_process = jax.jit(lambda t : jnp.transpose(t, (0, 1, 3, 4, 2)) / 255.0)
    #agent.obs_process = jax.jit(lambda t : t / 255.0)
    return agent

class FakeSyncEnv:
    def __init__(self, name, num_envs, batch_size):
        
        import envpool
        self.env = envpool.make_gym(name, num_envs=num_envs, batch_size=batch_size)
        self.batch_size = batch_size
        self.n = num_envs
        self.m = 1

        self.env.async_reset()

    def recv(self):
        obs, reward, done, info = self.env.recv()
        return obs, reward, done, info['env_id']

    def send(self, *args, **kargs):
        self.env.send(*args, **kargs)

    def close(self, *args, **kargs):
        self.env.close(*args, **kargs)

    def re_init(self, *args, **kargs):
        pass

if __name__ == "__main__":
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    import numpy as np
    from utils.conventions import *
    from jax.random import PRNGKey
    from utils.queue import Queue

    agent = make_agent()

    from envs.wrapper import make_env
    from envs.VecEnv import SyncEnv, AsyncEnv

    N_steps = 10

    env_m = 5
    env_n = 16
    batch_size = 32
    #env = AsyncEnv(env_fn, (84, 84, 4), np.uint8, env_n, env_m, batch_size, render=True)
    env = FakeSyncEnv("Breakout-v5", 96, 32)

    try:
        import threading
        from algorithms.Actor import *
        from algorithms.Learner import *
        from utils.SharedArray import *

        params = agent.init_params(PRNGKey(42))
        shared_params = FakeSharedJaxParams(params)

        learner_queue = Queue(128, is_mp_queue=False)

        process = [threading.Thread(target=workAsync, args=(env, make_agent, shared_params, learner_queue, WorkerArgs()), daemon=True) for _ in range(1)]
        [p.start() for p in process]

        learnAsync(shared_params, make_agent, learner_queue, agent.RETRACE_LOSS, HyperParams(), 3600)

    except KeyboardInterrupt:
        env.close()
