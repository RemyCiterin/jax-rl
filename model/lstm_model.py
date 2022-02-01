import jax 
import jax.numpy as jnp
import haiku as hk
import chex

from utils.conventions import *

class MLP_MODEL(hk.RNNCore):
    def __init__(self, outDim):
        self.outDim = outDim
        super().__init__()

    def initial_state(self, batch_size):
        return hk.LSTM(512).initial_state(batch_size)

    def __call__(self, obs : Observation, state_tm1):

        state_tm1 = jax.tree_map(
            lambda state : jax.vmap(lambda a, b : a * b)(state, 1-obs.done_tm1), 
            state_tm1
        )

        h, state = hk.LSTM(512)(obs.obs, state_tm1)

        h = hk.Sequential([
            hk.Linear(256), jax.nn.relu
        ])(h)

        return ModelOutput(
            value=jnp.squeeze(hk.Linear(1)(h), axis=-1),
            logits=hk.Linear(self.outDim)(h)
        ), state

class CONV_MODEL(MLP_MODEL):
    def __call__(self, obs : Observation, state_tm1):

        convModel = hk.Sequential([
            hk.Conv2D(32, 8, 4, padding='Valid'), jax.nn.relu, 
            hk.Conv2D(64, 4, 2, padding='Valid'), jax.nn.relu, 
            hk.Conv2D(64, 3, 1, padding='Valid'), jax.nn.relu
        ])

        h = jax.vmap(lambda o : 
            jnp.reshape(convModel(o), (-1,))
        )(obs.obs)

        return super().__call__(Observation(obs=h, done_tm1=obs.done_tm1), state_tm1)
