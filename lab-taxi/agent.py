import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.001 #0.005
        self.alpha = 0.9
        self.gamma = 1.0
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        #return np.random.choice(self.nA)

        # Select an epsilon-greedy action for the current state
        if np.random.random() > self.epsilon:
            # Select the greedy action
            return np.argmax(self.Q[state])
        else:
            # Select randomly an action
            return np.random.choice(np.arange(self.nA))
        
    
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.Q[state][action] += 1

        # Implementation using Temporal Difference method : Expected Sarsa

        # Backup current Q for state, action pair
        Qsa = self.Q[state][action]      
        
        # With probability epsilon, the agent will select an action uniformly 
        # at random from the set of available (non-greedy AND greedy) actions
        pi_s = np.ones(self.nA) * self.epsilon / self.nA

        # With probability (1 - epsilon), the agent will select the greedy action
        best_a = np.argmax(self.Q[next_state])
        pi_s[best_a] = (1 - self.epsilon) + (self.epsilon / self.nA)
      
        # Retrieve the expected value for the next_state (If end of episode / next_state is None, then return 0)
        Qsa_next = np.dot(self.Q[next_state], pi_s) if next_state is not None  else 0
        
        # Update the Q-Table
        self.Q[state][action] = Qsa + self.alpha * ((reward + self.gamma * Qsa_next) - Qsa)