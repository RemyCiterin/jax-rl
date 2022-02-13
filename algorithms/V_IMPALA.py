from algorithms.BaseAgent import BaseAgent
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

class Agent(BaseAgent):
    def __init__(self, core : Callable[[int], hk.RNNCore], inDim, 
            outDim, optim=INIT_OPTIM, prefix=0, use_H_target=False):

        self._init_fn, self._apply_fn = hk.without_apply_rng(
            hk.transform(lambda obs : core(outDim)(obs))
        )

        _, self._init_rnn = hk.without_apply_rng(
            hk.transform(lambda batch_size : core(outDim).initial_state(batch_size))
        )

        self.use_H_target = use_H_target
        self.prefix = prefix
        self.outDim = outDim
        self.inDim = inDim
        self.optim = optim
        self.core  = core

    @partial(jax.jit, static_argnums=(0, 1))
    def init_rnn(self, batch_size):
        return self._init_rnn(None, batch_size)

    @partial(jax.jit, static_argnums=0)
    def obs_process(self, obs):
        return obs


    @partial(jax.jit, static_argnums=(0,))
    def init_state(self, params):
        return self.optim.init(params)

    @partial(jax.jit, static_argnums=0)
    def init_params(self, rng):
        params = {"params": self._init_fn(rng, jnp.zeros(self.inDim)[None, None])}
        if self.use_H_target: params['H_coef'] = jnp.log(0.01)
        return params

    @partial(jax.jit, static_argnums=0)
    def get_action(self, rng, params, obs):
        model_output = jax.tree_map(lambda t : t[0],
            self._apply_fn(params['params'], self.obs_process(jax.tree_map(lambda t : t[None], obs))))
        softmax = jax.nn.softmax(model_output.logits)
        rng_, rng = jax.random.split(rng)

        actions = jax.vmap(
            lambda s, r : jax.random.choice(r, self.outDim, p=s))(
            softmax, jnp.stack(jax.random.split(rng, len(obs)), axis=0)
        )

        return actions, model_output.logits, rng_

    @partial(jax.jit, static_argnums=0)
    def V_TRACE_LOSS(self, result, tau:Tau, args:HyperParams, **kargs):
        T, B = tau.done.shape

        value = result.value
        logits = result.logits.at[:T].get()

        ln_mu = jnp.sum(
            jax.nn.log_softmax(tau.data) * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        ln_pi = jnp.sum(
            jax.nn.log_softmax(logits) * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        gamma = args.Gamma * (1 - tau.done.at[:T].get())
        IS = jnp.exp(ln_pi - ln_mu)

        def aux(v_tm1, v_t, r_t, discount_t, rho_tm1):
            return rlax.leaky_vtrace_td_error_and_advantage(
                v_tm1, v_t, r_t, discount_t, rho_tm1,
                lambda_=args.Lambda,
                alpha=args.IS_coef
            )

        error = jax.vmap(
            aux, in_axes=1, out_axes=1)(
            value.at[:T].get(), value.at[1:T+1].get(),
            tau.reward.at[:T].get(), gamma, sg(IS)
        )

        lossKL = args.KL_coef * jnp.sum(
            jax.nn.softmax(tau.data) * (
                jax.nn.log_softmax(tau.data) -
                jax.nn.log_softmax(logits)
            ), axis=-1
        )

        entropy = - jnp.sum(
            jax.nn.log_softmax(logits) *
            jax.nn.softmax(logits)
            , axis=-1
        )

        if self.use_H_target:
            lossH = - entropy * sg(kargs['H_coef']) + \
                sg(entropy - args.H_target * jnp.log(self.outDim)) * kargs['H_coef']
            
        else:
            lossH = - args.H_coef * entropy

        lossP = - args.P_coef * ln_pi * sg(error.pg_advantage)
        lossV = args.V_coef * error.errors ** 2

        return jnp.mean(lossP + lossKL + lossH + lossV)

    @partial(jax.jit, static_argnums=(0, 1))
    def update(self, error_loss, params, state, tau, args):

        def get_loss(params, tau, args):
            kargs = {}

            if self.use_H_target: kargs['H_coef'] = jnp.exp(params['H_coef'])

            model_output = self._apply_fn(params['params'], self.obs_process(tau.obs))
            return error_loss(
                jax.tree_map(lambda t : t.at[self.prefix:].get(), model_output), 
                jax.tree_map(lambda t : t.at[self.prefix:].get(), tau), args, **kargs)


        loss, grad = jax.value_and_grad(get_loss)(params, tau, args)

        updates, state = self.optim.update(grad, state, params)
        params = optax.apply_updates(params, updates)

        return state, params, loss
