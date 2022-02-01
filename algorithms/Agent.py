from utils.conventions import *
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import chex
import rlax
import jax

from typing import Callable
from functools import partial

INIT_OPTIM = optax.chain(
    optax.clip_by_global_norm(40.0), 
    optax.rmsprop(5e-4, decay=0.99)
)

class Agent:
    def __init__(self, core : Callable[[int], hk.RNNCore], inDim, outDim, optim=INIT_OPTIM):

        self._init_fn, self._apply_fn = hk.without_apply_rng(
            hk.transform(lambda obs, st : core(outDim)(obs, st))
        )

        _, self._unroll_fn = hk.without_apply_rng(
            hk.transform(lambda obs, st : hk.dynamic_unroll(core(outDim), obs, st))
        )

        _, self._init_rnn = hk.without_apply_rng(
            hk.transform(lambda batch_size : core(outDim).initial_state(batch_size))
        )

        self.outDim = outDim
        self.inDim = inDim
        self.optim = optim

    @partial(jax.jit, static_argnums=0)
    def obs_process(self, obs):
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def init_state(self, params):
        return self.optim.init(params)

    @partial(jax.jit, static_argnums=(0, 1))
    def init_rnn(self, batch_size):
        return self._init_rnn(None, batch_size)

    @partial(jax.jit, static_argnums=0)
    def init_params(self, rng):
        return self._init_fn(rng, Observation(obs=jnp.zeros(self.inDim)[None], done_tm1=jnp.zeros((1,))), self.init_rnn(1))

    @partial(jax.jit, static_argnums=0)
    def get_action(self, rng, params, obs : Observation, rnn_st):
        model_output, rnn_st = self._apply_fn(params, Observation(obs=self.obs_process(obs.obs), done_tm1=obs.done_tm1), rnn_st)
        softmax = jax.nn.softmax(model_output.logits)

        actions = jax.vmap(
            lambda s, r : jax.random.choice(r, self.outDim, p=s))(
            softmax, jnp.stack(jax.random.split(rng, len(obs.obs)), axis=0)
        )

        return actions, model_output.logits, rnn_st

    @partial(jax.jit, static_argnums=0)
    def V_TRACE_LOSS(self, params, tau:Tau, args:HyperParams):
        T, B = tau.done.shape

        model_output, _ = self._unroll_fn(
            params, Observation(obs=self.obs_process(tau.obs.obs), done_tm1=tau.obs.done_tm1), tau.rnn)
        value = model_output.value

        ln_mu = jnp.sum(
            jax.nn.log_softmax(tau.logits).at[:T].get() * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        ln_pi = jnp.sum(
            jax.nn.log_softmax(model_output.logits).at[:T].get() * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        gamma = args.Gamma * (1 - tau.done.at[:T].get())
        IS = jnp.exp(ln_pi - ln_mu)

        error = jax.vmap(
            partial(rlax.leaky_vtrace_td_error_and_advantage, lambda_=args.Lambda, alpha=args.IS_coef), in_axes=1
        )(
            v_tm1=value.at[1:T+1].get(), v_t=value.at[:T].get(), 
            r_t=tau.reward.at[:T].get(), discount_t=gamma, rho_tm1=IS, 
        )

        lossP = - args.P_coef * ln_pi * sg(error.pg_advantage)
        lossKL = args.KL_coef * jnp.exp(ln_mu) * ln_pi
        lossH = args.H_coef * ln_pi * jnp.exp(ln_pi)
        lossV = args.V_coef * error.errors ** 2

        return jnp.mean(lossP + lossKL + lossH + lossV)

    @partial(jax.jit, static_argnums=(0, 1))
    def update(self, error_loss, params, state, tau, args):
        loss, grad = jax.value_and_grad(error_loss)(params, tau, args)

        updates, state = self.optim.update(grad, state, params)
        params = optax.apply_updates(params, updates)

        return state, params, loss
