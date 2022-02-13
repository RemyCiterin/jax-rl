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
    def __init__(self, core : Callable[[int], hk.RNNCore], inDim, outDim, optim=INIT_OPTIM, 
            prefix=0, use_H_target=False):

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
        return self.optim.init(params[0])

    @partial(jax.jit, static_argnums=0)
    def init_params(self, rng):
        params = {}

        params['params'] = self._init_fn(rng, jnp.zeros(self.inDim)[None, None])
        if self.use_H_target: params['H_coef'] = jnp.log(0.01)
        params['adv_std'] = jnp.array(1.0)
        return (params, params)

    @partial(jax.jit, static_argnums=0)
    def get_action(self, rng, params, obs):
        model_output = jax.tree_map(lambda t : t[0], 
            self._apply_fn(params[0]['params'], self.obs_process(jax.tree_map(lambda t : t[None], obs))))
        softmax = jax.nn.softmax(model_output.logits)
        rng_, rng = jax.random.split(rng)

        actions = jax.vmap(
            lambda s, r : jax.random.choice(r, self.outDim, p=s))(
            softmax, jnp.stack(jax.random.split(rng, len(obs)), axis=0)
        )

        return actions, model_output.logits, rng_

    @partial(jax.jit, static_argnums=0)
    def RETRACE_LOSS(self, result, target, tau:Tau, args:HyperParams, **kargs):
        T, B = tau.done.shape

        logits = result.logits.at[:T].get()

        ln_mu = jnp.sum(
            jax.nn.log_softmax(tau.data) * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        ln_pi = jnp.sum(
            jax.nn.log_softmax(logits) * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        ln_pi_target = jnp.sum(
            jax.nn.log_softmax(target.logits.at[:T].get()) * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        Q = jnp.sum(result.value.at[:T].get() * jax.nn.one_hot(tau.action, self.outDim), axis=-1)

        gamma = args.Gamma * (1 - tau.done.at[:T].get())
        IS_target = jnp.exp(ln_pi_target - ln_mu)

        Qt = jnp.sum(target.value.at[:T].get() * jax.nn.one_hot(tau.action, self.outDim), axis=-1)
        Vt = jnp.sum(jax.nn.softmax(target.logits) * target.value, axis=-1).at[:T+1].get()

        C = jnp.minimum(1, sg(IS_target))

        g = tau.reward.at[T-1].get() + gamma.at[T-1].get() * Vt.at[T].get()
        G = [g]

        for i in reversed(range(T-1)):
            g = tau.reward.at[i].get() + gamma.at[i].get() * (
                Vt.at[i+1].get() + C.at[i+1].get() * (g - Qt.at[i+1].get()))
            G.insert(0, g)
        G = jnp.array(G)

        lossKL = args.KL_coef * jnp.sum(
            jax.nn.softmax(logits).at[:T].get() * (
                jax.nn.log_softmax(logits).at[:T].get() -
                jax.nn.log_softmax(target.logits).at[:T].get()
            ), axis=-1
        )

        entropy = - jnp.sum(
            jax.nn.softmax(logits) *
            jax.nn.log_softmax(logits)
            , axis=-1
        )

        if self.use_H_target:
            lossH = - entropy * sg(jnp.exp(kargs['H_coef'])) + \
                sg(entropy - args.H_target * jnp.log(self.outDim)) * jnp.exp(kargs['H_coef'])
            
        else:
            lossH = - args.H_coef * entropy

        adv = sg(G - Vt.at[:T].get())
        lossA = (kargs['adv_std'] - adv ** 2) ** 2
        adv = adv / sg(kargs['adv_std']+1e-7) ** 0.5

        lossP = - args.P_coef * jnp.exp(ln_pi - ln_mu) * adv
        lossV = args.V_coef * (Q - sg(G)) ** 2

        return jnp.mean(lossP + lossKL + lossH + lossV + lossA)

    @partial(jax.jit, static_argnums=0)
    def MUESLI_LOSS(self, result, target, tau:Tau, args:HyperParams, **kargs):
        T, B = tau.done.shape

        logits = result.logits.at[:T].get()

        ln_mu = jnp.sum(
            jax.nn.log_softmax(tau.data) * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        ln_pi = jnp.sum(
            jax.nn.log_softmax(logits) * jax.nn.one_hot(tau.action, self.outDim)
            , axis=-1
        )

        Q = jnp.sum(result.value.at[:T].get() * jax.nn.one_hot(tau.action, self.outDim), axis=-1)

        pi_prior = jax.nn.softmax(target.logits)

        if self.use_H_target:
            pi_prior = sg(
                pi_prior * (1-jax.nn.sigmoid(kargs['H_coef'])) + \
                jax.nn.sigmoid(kargs['H_coef']) * jnp.ones_like(target.logits) / self.outDim
            )

        pi_prior_by_act = jnp.sum(pi_prior.at[:T].get() * \
            jax.nn.one_hot(tau.action, self.outDim), axis=-1)

        gamma = args.Gamma * (1 - tau.done.at[:T].get())
        IS_target = pi_prior_by_act / jnp.exp(ln_mu)

        Qt = jnp.sum(target.value.at[:T].get() * jax.nn.one_hot(tau.action, self.outDim), axis=-1)
        Vt = jnp.sum(pi_prior * target.value, axis=-1).at[:T+1].get()

        C = jnp.minimum(1, sg(IS_target))

        g = tau.reward.at[T-1].get() + gamma.at[T-1].get() * Vt.at[T].get()
        G = [g]

        for i in reversed(range(T-1)):
            g = tau.reward.at[i].get() + gamma.at[i].get() * (
                Vt.at[i+1].get() + C.at[i+1].get() * (g - Qt.at[i+1].get()))
            G.insert(0, g)
        G = jnp.array(G)

        adv = sg(G - Vt.at[:T].get())
        lossA = (kargs['adv_std'] - adv ** 2) ** 2
        adv = adv / sg(kargs['adv_std']+1e-7) ** 0.5

        pi_CMPO = pi_prior.at[:T].get() * jnp.exp(jnp.clip(
            (target.value.at[:T].get() - Vt.at[:T].get()[..., None]) / \
            sg(kargs['adv_std']+1e-7) ** 0.5, -1.0, 1.0))

        pi_CMPO = pi_CMPO / jnp.sum(pi_CMPO, axis=-1, keepdims=True)

        lossKL = - args.KL_coef * jnp.sum(
            sg(pi_CMPO) * jax.nn.log_softmax(logits),
            axis=-1
        )

        entropy = - jnp.sum(
            jax.nn.log_softmax(logits) *
            jax.nn.softmax(logits)
            , axis=-1
        )

        if self.use_H_target:
            lossH = sg(entropy - args.H_target * jnp.log(self.outDim)) * kargs['H_coef']
            
        else:
            lossH = - args.H_coef * entropy

        lossP = - args.P_coef * jnp.exp(ln_pi - ln_mu) * adv
        lossV = args.V_coef * (Q - sg(G)) ** 2

        return jnp.mean(lossP + lossH + lossKL + lossV + lossA)

    @partial(jax.jit, static_argnums=(0, 1))
    def update(self, error_loss, params, state, tau, args):
        target  = params[1]
        params  = params[0]

        def get_loss(params, target, tau, args):
            kargs = {'adv_std' : params['adv_std']}

            if self.use_H_target: kargs['H_coef'] = params['H_coef']
            model_output  = self._apply_fn(params['params'], self.obs_process(tau.obs))
            target_output = self._apply_fn(target['params'], self.obs_process(tau.obs))
            return error_loss(
                jax.tree_map(lambda t : t.at[self.prefix:].get(), model_output),
                jax.tree_map(lambda t : t.at[self.prefix:].get(), target_output),
                jax.tree_map(lambda t : t.at[self.prefix:].get(), tau),
                args, **kargs)

        loss, grad = jax.value_and_grad(get_loss)(params, target, tau, args)

        updates, state = self.optim.update(grad, state, params)
        params = optax.apply_updates(params, updates)

        target = jax.tree_map(lambda x, y : 0.1*x+0.9*y, params, target)

        return state, (params, target), loss
