import os 
os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "false"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

import numpy as np

from utils.conventions import *
from algorithms.Agent import Agent
from model.lstm_model import CONV_MODEL
from jax.random import PRNGKey, split
import jax.numpy as jnp

from functools import partial


from algorithms.PartialTau import PartialTau

agent = Agent(CONV_MODEL, (84, 84 ,4), 4)
agent.obs_process = jax.jit(lambda t : t / 255.0)

params = agent.init_params(PRNGKey(42))
opti_state = agent.init_state(params)

from envs.wrapper import make_env
from envs.VecEnv import SyncEnv

env_fn = lambda : make_env("BreakoutNoFrameskip-v4")

N_steps = 10

env_m = 2
env_n = 16
batch_size = 32
env = SyncEnv(env_fn, (84, 84, 4), np.uint8, env_n, env_m, batch_size)

current_r_sum = jnp.zeros((env_n * env_m, batch_size))
last_r_sum    = jnp.zeros((env_n * env_m, batch_size))

@partial(jax.jit, backend="cpu")
def update_recoder(idx, reward, done, c_sum, l_sum):
    return (
        c_sum.at[idx].set((c_sum.at[idx].get() + reward) * (1 - done)), 
        l_sum.at[idx].set((c_sum.at[idx].get() + reward) * done + l_sum.at[idx].get() * (1-done))
    )

partial_tau = [PartialTau(N_steps, use_ETD=False) for _ in range(env_n * env_m)]
rnn_state  = [agent.init_rnn(batch_size) for _ in range(env_n*env_m)]
prev_state = [None for _ in range(env_n*env_m)]

steps = 0
rng = PRNGKey(42)

while True:
    obs, reward, done, ident = env.recv()
    obs = Observation(obs=obs, done_tm1=done)

    rng, _rng = split(rng)
    actions, logits, rnn = to_np(agent.get_action(_rng, params, obs, rnn_state[ident]))
    env.send(actions.astype(np.int64), ident)

    current_r_sum, last_r_sum = update_recoder(ident, reward, done, current_r_sum, last_r_sum)

    if not prev_state[ident] is None:
        p_obs, p_rnn, p_actions, p_logits = prev_state[ident]
        tau = partial_tau[ident].add_transition(p_obs, p_rnn, p_logits, p_actions, np.clip(reward, -1, 1), done, obs, rnn)
        if not tau is None: opti_state, param, _ = agent.update(agent.V_TRACE_LOSS, params, opti_state, tau, HyperParams())


    prev_state[ident] = obs, rnn_state[ident], actions, logits
    rnn_state[ident] = rnn


    steps += 1

    if steps % 100 == 0:
        print(end=f"\r{steps}   {np.mean(last_r_sum)}   ")