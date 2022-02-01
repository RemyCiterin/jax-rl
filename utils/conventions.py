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

class Tau:
    obs : Observation
    done : chex.Array
    reward : chex.Array
    action : chex.Array
    logits : chex.Array

@chex.dataclass
class HyperParams:
    alpha_is : chex.array
    lambda_ : chex.array
    KL_coef : chex.Array
    H_coef : chex.Array
    P_coef : chex.Array
    V_coef : chex.Array
    gamma : chex.Array


def to_np(params):
    return jax.tree_map(np.array, params)

def to_jnp(params):
    return jax.tree_map(jnp.array, params)