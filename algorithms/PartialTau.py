import numpy as np
import jax

from utils.conventions import *
class PartialTau:
    def __init__(self, trajectory_n, use_ETD=False):
        self.use_ETD = use_ETD
        self.n = trajectory_n
        self.tau = None
    
    def add_transition_V_TRACE(self, obs, logits, action, reward, done, n_obs):
        if self.tau is None: self.tau = Tau(obs=[obs], action=[], reward=[], done=[], data=[])

        self.tau.obs.append(n_obs)
        self.tau.done.append(done)
        self.tau.data.append(logits)
        self.tau.action.append(action)
        self.tau.reward.append(reward)

        if len(self.tau.done) == self.n:


            tau = Tau(
                done  =np.array(self.tau.done),
                reward=np.array(self.tau.reward),
                obs   =jax.tree_multimap(lambda *args: np.array(args), *self.tau.obs),
                data  =jax.tree_multimap(lambda *args: np.array(args), *self.tau.data),
                action=jax.tree_multimap(lambda *args: np.array(args), *self.tau.action),
            )

            self.tau = Tau(
                obs   =[n_obs],
                done  =[],
                reward=[],
                action=[],
                data  =[],
            )
            return tau 
        return None

    def add_transition_ETD(self, obs, logits, action, reward, done, n_obs):
        if self.tau is None: self.tau = Tau(obs=[obs], action=[], reward=[], done=[], data=[])
        
        self.tau.obs.append(n_obs)
        self.tau.done.append(done)
        self.tau.data.append(logits)
        self.tau.action.append(action)
        self.tau.reward.append(reward)
        

        if len(self.tau.obs) == 2*self.n:
            tau = Tau(
                done  =np.array(self.tau.done),
                reward=np.array(self.tau.reward),
                obs   =jax.tree_multimap(lambda *args: np.array(args), *self.tau.obs),
                data  =jax.tree_multimap(lambda *args: np.array(args), *self.tau.data),
                action=jax.tree_multimap(lambda *args: np.array(args), *self.tau.action),
            )

            self.tau = Tau(
                obs   =self.tau.obs[self.n:2*self.n], 
                data  =self.tau.data[self.n:2*self.n-1],
                done  =self.tau.done[self.n:2*self.n-1], 
                reward=self.tau.reward[self.n:2*self.n-1], 
                action=self.tau.action[self.n:2*self.n-1], 
            )

            return tau 
        return None


    def add_transition(self, obs, logits, action, reward, done, n_obs):
        if self.use_ETD: return self.add_transition_ETD(obs, logits, action, reward, done, n_obs)
        return self.add_transition_V_TRACE(obs, logits, action, reward, done, n_obs)
