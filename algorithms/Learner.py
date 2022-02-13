from utils.conventions import *
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import time
import chex
import rlax
import jax

from typing import Callable
from functools import partial

def learnSync(shared_params, agent_fn, learner_queue, loss_fn, hyper_params, total_time, update_params_freq=10):
    agent = agent_fn()

    shared_params.re_init()
    params = to_jnp(shared_params.get())
    opti_state = agent.init_state(params)

    print("learner start !!!")

    start_time = time.time()
    steps = 0

    while time.time() - start_time < total_time:
        tau = learner_queue.get(timeout=60)
        if steps % update_params_freq == 0: shared_params.set(params)
        opti_state, params, _ = agent.update(loss_fn, params, opti_state, tau, hyper_params)
        steps += 1

def learnAsync(shared_params, agent_fn, learner_queue, loss_fn, hyper_params, total_time, update_params_freq=10, batch_size=32):
    agent = agent_fn()

    shared_params.re_init()
    params = to_jnp(shared_params.get())
    opti_state = agent.init_state(params)

    print("learner start !!!")

    start_time = time.time()
    steps = 0

    while time.time() - start_time < total_time:
        if steps % update_params_freq == 0: shared_params.set(params)
        tau = [learner_queue.get(timeout=60) for _ in range(batch_size)]
        tau = jax.tree_multimap(lambda *args: np.stack(args, axis=1), *tau)
        opti_state, params, _ = agent.update(loss_fn, params, opti_state, tau, hyper_params)
        steps += 1