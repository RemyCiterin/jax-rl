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
    def update_state(self, G, action, index, reward):

        I = jax.lax.cond(
            jnp.less(self.I.at[action].get(), 0),
            lambda _ : self.I.at[action].set(index),
            lambda _ : self.I, None)

        R = jax.lax.cond(
            jnp.less(self.I.at[action].get(), 0), 
            lambda _ : self.R.at[action].set(reward), 
            lambda _ : self.R, None)

        return NODE_STATE(S=self.S, P=self.P, R=R, I=I, V=self.V, 
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
        
        index_list = -jnp.ones((max_depth+1,), dtype=int)
        action_list = jnp.ones((max_depth+1,), dtype=int)
        value_list = jnp.zeros((max_depth+1,))

        def body_fun(state):
            index_list, action_list, value_list, index, i = state

            node = jax.tree_map(lambda t : t.at[index].get(), self.Nodes)
            action = node.get_action(params)

            return index_list.at[i].set(index), action_list.at[i].set(action),\
                value_list.at[i].set(node.R.at[action].get()), node.I.at[action].get(), i+1

        def cond_fun(state):
            _, _, _, index, i = state
            return jnp.logical_and(jnp.less(i, max_depth+1), jnp.greater_equal(index, 0))

        index_list, action_list, value_list, _, i_p1 = jax.lax.while_loop(
            cond_fun, body_fun, (index_list, action_list, value_list, index, jnp.array(0, dtype=int)))

        index, action = index_list.at[i_p1-1].get(), action_list.at[i_p1-1].get()
        node = jax.tree_map(lambda t : t.at[index].get(), self.Nodes)

        S, r = model.dynamics(node.S, action)
        model_output = model.evaluation(S)
        P = jax.nn.softmax(model_output.logits)

        new_node = NODE_STATE.init_node(S=S, P=P, V=model_output.value)
        value_list = value_list.at[i_p1-1].set(r+params.gamma*new_node.V)

        for i in range(len(value_list)-2, -1, -1):
            value_list = value_list.at[i].add(params.gamma * value_list.at[i+1].get())

        return index_list, value_list, action_list, new_node, r, max_depth+1-i_p1

    @partial(jax.jit)
    def update(self, index_list, value_list, action_list, new_node, last_reward, final_depth):
        new_index_or_minus_one = len(self.Nodes.V) * jnp.greater(final_depth, 0) - jnp.equal(final_depth, 0)

        def body_fun(state):
            i, tree = state

            node = jax.tree_map(lambda t : t.at[index_list.at[i].get()].get(), tree.Nodes)
            update_node = node.update_state(value_list.at[i].get(), action_list.at[i].get(), 
                new_index_or_minus_one, last_reward)

            return i+1, MCTS_TREE(
                Nodes=jax.tree_map(lambda x, y : x.at[index_list.at[i].get()].set(y), tree.Nodes, update_node))

        def cond_fun(state):
            i, tree = state

            return jnp.logical_and(
                jnp.less(i, len(index_list)),
                jnp.greater_equal(index_list.at[i].get(), 0))
        
        tree = jax.lax.while_loop(cond_fun, body_fun, (jnp.array(0, dtype=int), self))[1]
        return MCTS_TREE(Nodes=jax.tree_map(lambda x, y : jnp.concatenate((x, y[None]), axis=0), tree.Nodes, new_node))

    @partial(jax.jit, static_argnames=('num_sim', 'max_depth', 'model'))
    def mcts_step(self, rng, model, params, max_depth, num_sim):
        for _ in range(num_sim):
            rng, rng_ = jax.random.split(rng)
            index_list, value_list, action_list, new_node, last_reward, final_depth = self.simulate(
                rng_, jnp.array(0, dtype=int), model, params, max_depth)

            self = self.update(index_list, value_list, action_list, new_node, last_reward, final_depth)

        return self


import time


rng = PRNGKey(43)
mask = jnp.ones((3,))
obs = jnp.zeros((3,))

model = MODEL(
    representation=lambda o : o,
    evaluation=lambda o : ModelOutput(logits=o, value=jnp.array(7.0)),
    dynamics=lambda o, a : (o, jnp.array(0.1))
)

params = PARAMS()

"""

@partial(jax.jit, static_argnames=('model', 'noise'))
def make_tree(rng, obs, mask, model, params, noise):
    return jax.vmap(lambda _ : MCTS_TREE.init_tree(rng, obs, mask, model, params, noise))(jnp.arange(1024))

@partial(jax.jit, static_argnames=('num_sim', 'max_depth', 'model'))
def make_step(tree, rng_, model, params, max_depth, num_sim):
    return jax.vmap(lambda t : t.mcts_step(rng_, model, params, max_depth, num_sim))(tree)

for _ in range(10):
    t = time.time()
    print("-----------------")
    rng, rng_ = jax.random.split(rng)
    tree = make_tree(rng_, obs, mask, model, params, 'dirichlet')

    tree = make_step(tree, rng_, model, params, 10, 50)
    jax.tree_map(lambda t : t.block_until_ready(), tree)
    print(time.time()-t)

print(jax.tree_map(lambda t : t.shape, tree))

"""

tree = MCTS_TREE.init_tree(rng, obs, mask, model, params, "dirichlet")
print(tree)


index_list, value_list, action_list, new_node, last_reward, final_depth = tree.simulate(
    rng, jnp.array(0, dtype=int), model, params, 0)

print(index_list)
print(value_list)
print(action_list)
print(new_node)
print(last_reward)
print(final_depth)


tree = tree.update(index_list, value_list, action_list, new_node, last_reward, final_depth)

print(tree)