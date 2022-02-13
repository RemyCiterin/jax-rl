from functools import partial
import jax

class BaseAgent(object):
    def __init__(self, core, inDim, outDim, **kargs):
        self.outDim = outDim
        self.inDim = inDim

    def init_rnn(self, batch_size):
        return None

    def obs_process(self, obs):
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def init_state(self, params):
        return None

    def init_params(self, rng):
        return {}

    @partial(jax.jit, static_argnums=0)
    def get_action(self, rng, params, obs):
        rng, rng_ = jax.random.split(rng)
        actions = jax.random.choice(rng, self.outDim, size=len(obs))
        return actions, None, rng_

    def update(self, *args, **kargs):
        return None, None, 0.0

class AgentWrapper(object):
    def __init__(self, agent):
        self.agent = agent

    def init_rnn(self, *args, **kargs):
        self.agent.init_rnn(*args, **kargs)

    def obs_process(self, *args, **kargs):
        return self.agent.obs_process(*args, **kargs)

    def init_state(self, *args, **kargs):
        return self.agent.init_state(*args, **kargs)

    def init_params(self, *args, **kargs):
        return self.agent.init_params(*args, **kargs)

    def get_action(self, *args, **kargs):
        return self.agent.get_action(*args, **kargs)
        
    def update(self, *args, **kargs):
        return self.agent.update(*args, **kargs)