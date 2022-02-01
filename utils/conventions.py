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
class Observation:
    obs : chex.Array
    done_tm1 : chex.Array

@chex.dataclass
class Tau:
    obs : Observation
    done : chex.Array
    reward : chex.Array
    action : chex.Array
    logits : chex.Array
    rnn : chex.ArrayTree

@chex.dataclass
class HyperParams:
    V_coef  : chex.Array = 1.0
    P_coef  : chex.Array = 1.0
    IS_coef : chex.Array = 1.0
    KL_coef : chex.Array = 5.0
    Gamma   : chex.Array = 0.99
    Lambda  : chex.Array = 0.95
    H_coef  : chex.Array = 0.01


def to_np(params):
    return jax.tree_map(np.asarray, params)

def to_jnp(params):
    return jax.tree_map(jnp.array, params)