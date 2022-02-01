import numpy as np
import jax

from utils.conventions import *
class PartialTau:
    def __init__(self, trajectory_n, use_ETD=False):
        self.use_ETD = use_ETD
        self.n = trajectory_n
        self.tau = None
    
    def add_transition_V_TRACE(self, obs, rnn, logits, action, reward, done, n_obs, n_rnn):
        if self.tau is None: self.tau = Tau(obs=[obs], rnn=rnn, action=[], reward=[], done=[], logits=[])

        self.tau.obs.append(n_obs)
        self.tau.done.append(done)
        self.tau.action.append(action)
        self.tau.reward.append(reward)
        self.tau.logits.append(logits)

        if len(self.tau.done) == self.n:


            tau = Tau(
            	rnn=self.tau.rnn,
                done  =np.array(self.tau.done),
                reward=np.array(self.tau.reward),
                action=np.array(self.tau.action),
                logits=np.array(self.tau.logits),
                obs   =jax.tree_multimap(lambda *args: np.array(args), *self.tau.obs),
            )

            self.tau = Tau(
                obs   =[n_obs],
                rnn   =n_rnn,
                done  =[], 
                reward=[], 
                action=[], 
                logits=[], 
            )
            return tau 
        return None

    def add_transition_ETD(self, obs, rnn, logits, action, reward, done, n_obs, n_rnn):
        if self.tau is None: self.tau = Tau(obs=[obs], rnn=[rnn], action=[], reward=[], done=[], logits=[])
        
        self.tau.rnn.append(n_rnn)
        self.tau.obs.append(n_obs)
        self.tau.done.append(done)
        self.tau.action.append(action)
        self.tau.reward.append(reward)
        self.tau.logits.append(logits)
        

        if len(self.tau.obs) == 2*self.n:
            tau = Tau(
            	rnn   =self.tau.rnn[0],
                done  =np.array(self.tau.done),
                reward=np.array(self.tau.reward),
                action=np.array(self.tau.action),
                logits=np.array(self.tau.logits),
                obs   =jax.tree_multimap(lambda *args: np.array(args), *self.tau.obs),
            )

            self.tau = Tau(
            	rnn   =self.tau.rnn[self.n:2*self.n], 
                obs   =self.tau.obs[self.n:2*self.n], 
                done  =self.tau.done[self.n:2*self.n-1], 
                reward=self.tau.reward[self.n:2*self.n-1], 
                action=self.tau.action[self.n:2*self.n-1], 
                logits=self.tau.logits[self.n:2*self.n-1],
            )

            return tau 
        return None


    def add_transition(self, obs, rnn, logits, action, reward, done, n_obs, n_rnn):
        if self.use_ETD: return self.add_transition_ETD(obs, rnn, logits, action, reward, done, n_obs, n_rnn)
        return self.add_transition_V_TRACE(obs, rnn, logits, action, reward, done, n_obs, n_rnn)
