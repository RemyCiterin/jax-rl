from jax.random import PRNGKey, split
from utils.conventions import *
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import time
import chex
import rlax
import jax
import os

from functools import partial
from typing import Callable, NamedTuple
from algorithms.PartialTau import PartialTau

import tensorboardX


class WorkerArgs(NamedTuple):
    update_params_freq : int = 10
    queue_timeout : float = 60
    max_queue_size : int = 10
    total_time : int = 3600
    N_steps : int = 10


def workSync(envs, agent_fn, shared_params, learner_queue, args):
    agent = agent_fn()

    shared_params.re_init()
    params = shared_params.get()
    envs.re_init()

    print("actor start !!!")

    writer = tensorboardX.SummaryWriter()

    env_n, env_m, batch_size = envs.n, envs.m, envs.batch_size

    current_r_sum = jnp.zeros((env_n * env_m, batch_size))
    last_r_sum    = jnp.zeros((env_n * env_m, batch_size))

    @partial(jax.jit, backend="cpu")
    def update_recoder(idx, reward, done, c_sum, l_sum):
        return (
            c_sum.at[idx].set((c_sum.at[idx].get() + reward) * (1 - done)), 
            l_sum.at[idx].set((c_sum.at[idx].get() + reward) * done + l_sum.at[idx].get() * (1-done))
        )


    partial_tau = [PartialTau(args.N_steps, use_ETD=False) for _ in range(env_n * env_m)]
    prev_state = [None for _ in range(env_n*env_m)]

    steps = 0
    rng = PRNGKey(42)
    start_time = time.time()


    while time.time() - start_time < args.total_time:
        obs, reward, done, ident = envs.recv()

        if steps % args.update_params_freq == 0: params = shared_params.get()

        actions, logits, rng = to_np(agent.get_action(rng, params, obs))
        envs.send(actions.astype(np.int64), ident)

        current_r_sum, last_r_sum = update_recoder(ident, reward, done, current_r_sum, last_r_sum)

        if not prev_state[ident] is None:
            p_obs, p_actions, p_logits = prev_state[ident]
            tau = partial_tau[ident].add_transition(p_obs, p_logits, p_actions, np.clip(reward, -1, 1), done, obs)
            if not tau is None: learner_queue.put(tau, timeout=args.queue_timeout)


        prev_state[ident] = obs, actions, logits

        steps += 1

        if steps % 100 == 0:
            print(end=f"\r{steps*batch_size}   {np.mean(last_r_sum):.5f}   {int(time.time()-start_time)}      ")
            writer.add_scalar("reward", np.mean(last_r_sum), steps * batch_size)
            writer.flush()


def workAsync(envs, agent_fn, shared_params, learner_queue, args):
    agent = agent_fn()

    shared_params.re_init()
    params = shared_params.get()
    envs.re_init()

    print("actor start !!!")

    writer = tensorboardX.SummaryWriter()

    num_envs, batch_size = envs.n * envs.m, envs.batch_size

    current_r_sum = jnp.zeros((num_envs,))
    last_r_sum    = jnp.zeros((num_envs,))

    @partial(jax.jit, backend="cpu")
    def update_recoder(idx, reward, done, c_sum, l_sum):
        return (
            c_sum.at[idx].set((c_sum.at[idx].get() + reward) * (1 - done)), 
            l_sum.at[idx].set((c_sum.at[idx].get() + reward) * done + l_sum.at[idx].get() * (1-done))
        )

    partial_tau = [PartialTau(args.N_steps, use_ETD=False) for _ in range(num_envs)]
    prev_state = [None for _ in range(num_envs)]

    steps = 0
    rng = PRNGKey(42)
    start_time = time.time()

    while time.time() - start_time < args.total_time:
        obs, reward, done, ident_list = envs.recv()

        if steps % args.update_params_freq == 0: params = shared_params.get()

        actions, logits, rng = to_np(agent.get_action(rng, params, obs))
        envs.send(actions.astype(np.int64), ident_list)

        current_r_sum, last_r_sum = update_recoder(ident_list, reward, done, current_r_sum, last_r_sum)

        for i, ident in enumerate(ident_list):

            if not prev_state[ident] is None:
                p_obs, p_actions, p_logits = prev_state[ident]
                tau = partial_tau[ident].add_transition(p_obs, p_logits, p_actions, np.clip(reward[i], -1, 1), done[i], obs[i])
                if not tau is None: learner_queue.put(tau, timeout=args.queue_timeout)

            prev_state[ident] = obs[i], actions[i], logits[i]

        steps += 1

        if steps % 100 == 0:
            entropy = - jax.nn.softmax(logits) * jax.nn.log_softmax(logits) * agent.outDim
            print(end=f"\r{steps*batch_size}   {np.mean(last_r_sum):.5f}   {int(time.time()-start_time)}   {jnp.mean(entropy):5f}   ")
            writer.add_scalar("reward", np.mean(last_r_sum), steps * batch_size)
            writer.add_scalar("entropy", np.mean(entropy), steps * batch_size)
            writer.flush()