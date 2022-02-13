import jax 
import jax.numpy as jnp
import haiku as hk
import chex

from utils.conventions import *

class MLP_MODEL(hk.Module):
    def __init__(self, outDim):
        self.outDim = outDim
        super().__init__()

    def __call__(self, obs):

        h = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.Linear(256), jax.nn.relu
        ])(obs)

        return ModelOutput(
            value=jnp.squeeze(hk.Linear(1)(h), axis=-1),
            logits=hk.Linear(self.outDim)(h)
        )

class CONV_MODEL(MLP_MODEL):
    def __call__(self, obs):

        convModel = hk.Sequential([
            hk.Conv2D(32, 8, 4, padding='Valid'), jax.nn.relu,
            hk.Conv2D(64, 4, 2, padding='Valid'), jax.nn.relu,
            hk.Conv2D(64, 3, 1, padding='Valid'), jax.nn.relu
        ])

        h = jax.vmap(jax.vmap(lambda o :
            jnp.reshape(convModel(o), (-1,))
        ))(obs)

        return super().__call__(h)
