from jax.random import PRNGKey, split
import jax.numpy as jnp
import haiku as hk
import numpy as np
import chex
import jax 

from functools import partial
from typing import List, Tuple, Callable
from utils.conventions import *


@chex.dataclass
class PARAMS:
    max_depth : int = 5
    c1 : chex.Array = 1.25
    c2 : chex.Array = 19652.0
    gamma : chex.Array = 0.99
    noise_eps : chex.Array = 0.25
    noise_alpha : chex.Array = 0.03

@chex.dataclass
class NODE_STATE:
    S : chex.Array
    Q : chex.Array
    P : chex.Array
    N : chex.Array
    I : chex.Array
    R : chex.Array
    V : chex.Array

    @partial(jax.jit)
    def get_action(self, params:PARAMS):
        UCB = self.Q + self.P * jnp.sqrt(jnp.sum(self.N)) / (self.N+1) * \
            (params.c1 + jnp.log((jnp.sum(self.N)+params.c2+1) / params.c2))
        return jnp.argmax(UCB)

    @partial(jax.jit)
    def get_ucb(self, params:PARAMS):
        UCB = self.Q + self.P * jnp.sqrt(jnp.sum(self.N)) / (self.N+1) * \
            (params.c1 + jnp.log((jnp.sum(self.N)+params.c2+1) / params.c2))
        return UCB

    @partial(jax.jit)
    def update_state(self, G, action, index):

        I = jax.lax.cond(
            jnp.less(self.I.at[action].get(), 0), 
            lambda _ : self.I.at[action].set(index), 
            lambda _ : self.I, None)

        return NODE_STATE(S=self.S, P=self.P, R=self.R, I=I, V=self.V, 
            Q=self.Q+jax.nn.one_hot(action, len(self.N)) * (G-self.Q) / (self.N+1),
            N=self.N+jax.nn.one_hot(action, len(self.N)))

    @partial(jax.jit)
    def init_node(S, P, V):
        return NODE_STATE(S=S, P=P, V=V,
            R=jnp.zeros_like(P), I=-jnp.ones_like(P, dtype=int),
            Q=jnp.zeros_like(P), N=jnp.zeros_like(P, dtype=int))

@chex.dataclass
class MODEL:
    dynamics : Callable[[chex.ArrayTree, chex.Array], Tuple[chex.ArrayTree, chex.Array]]
    representation : Callable[[chex.ArrayTree], chex.Array]
    evaluation : Callable[[chex.ArrayTree], ModelOutput]

    def __hash__(self):
        return format(self).__hash__()

@chex.dataclass
class MCTS_TREE:
    Nodes : NODE_STATE

    @partial(jax.jit, static_argnames=('model', 'noise'))
    def init_tree(rng:chex.Array, obs:chex.Array, mask:chex.Array, model:MODEL, params:PARAMS, noise=None):

        S = model.representation(obs)
        model_output = model.evaluation(S)
        P = jax.nn.softmax(model_output.logits)

        if not noise is None and noise.lower() == "dirichlet":
            P = (1-params.noise_eps) * P + params.noise_eps * \
                jax.random.dirichlet(rng, params.noise_alpha*mask)

        if not noise is None and noise.lower() == "uniform":
            P = (1-params.noise_eps) * P + params.noise_eps * mask / jnp.sum(mask)

        return MCTS_TREE(Nodes=jax.tree_map(lambda t : t[None],
            NODE_STATE.init_node(S=S, P=P, V=model_output.value)
        ))

    @partial(jax.jit, static_argnames=('model', 'max_depth'))
    def simulate(self, rng:chex.Array, index:chex.Array, model:MODEL, params:PARAMS, max_depth:int):
        
        node = jax.tree_map(lambda t : t.at[index].get(), self.Nodes)
        action = node.get_action(params)

        def leaf_case(_):
            index_list = -jnp.ones((max_depth+1,))
            value_list = jnp.zeros((max_depth+1,))
            action_list = jnp.ones((max_depth+1,))


            S, r = model.dynamics(node.S, action)
            model_output = model.evaluation(S)
            P = jax.nn.softmax(
                model_output.logits)

            new_node = NODE_STATE.init_node(S=S, P=P, V=model_output.value)

            return index_list.at[0].set(index), value_list.at[0].set(r+params.gamma*new_node.V), \
                action_list.at[0].set(action), new_node, r

        def node_case(_):
            index_list, value_list, action_list, new_node, r = self.simulate(rng, index, model, params, max_depth-1)

            value = node.R.at[action].get() + params.gamma * value_list.at[0].get()

            return jnp.concatenate((index[None], index_list), axis=0), \
                jnp.concatenate((value[None], value_list), axis=0), \
                jnp.concatenate((action[None], action_list), axis=0), new_node, r

        if max_depth <= 0: return leaf_case(None)

        #print(
        #    jax.tree_map(lambda t : t.shape, leaf_case(None)),
        #    jax.tree_map(lambda t : t.shape, node_case(None)))

        return jax.lax.cond(
            jnp.less(node.I.at[action].get(), 0),
            leaf_case, node_case, None)

        @partial(jax.jit)
    def update(self, index_list, value_list, action_list, new_node, last_reward):
        pass

        


rng = PRNGKey(43)
mask = jnp.ones((81,))
obs = jnp.zeros((81,))

model = MODEL(
    representation=lambda o : o,
    evaluation=lambda o : ModelOutput(logits=o, value=jnp.array(7.0)),
    dynamics=lambda o, a : (o, jnp.array(0.1))
)

params = PARAMS()


tree = MCTS_TREE.init_tree(rng, obs, mask, model, params, noise='dirichlet')

index, value, actions, node, r = tree.simulate(PRNGKey(42), 0, model, params, 5)

print(actions)
print(index)
print(value)
print(node)
print(r)