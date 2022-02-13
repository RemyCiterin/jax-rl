import jax

import jax.numpy as jnp

import numpy as np

import chex

import jax

sg = jax.lax.stop_gradient

@chex.dataclass
class ModelOutput:
    logits:chex.Array
    value:chex.Array

@chex.dataclass
class ObsRNN:
    rnn: chex.ArrayTree
    obs: chex.ArrayTree
    is_reset: chex.Array

@chex.dataclass
class Tau:
    done : chex.Array
    reward : chex.Array
    action : chex.Array
    obs : chex.ArrayTree
    data : chex.ArrayTree

@chex.dataclass
class HyperParams:
    V_coef   : chex.Array = 1.0
    P_coef   : chex.Array = 1.0
    IS_coef  : chex.Array = 1.0
    KL_coef  : chex.Array = 1.0
    Lambda   : chex.Array = 1.0
    H_target : chex.Array = 0.8
    H_coef   : chex.Array = 0.01
    Gamma    : chex.Array = 0.99


def to_np(params):
    return jax.tree_map(np.asarray, params)

def to_jnp(params):
    return jax.tree_map(jnp.array, params)