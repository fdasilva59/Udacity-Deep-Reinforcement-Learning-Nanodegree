import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters  import *


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_dim, output_dim, seed=10, fc1_units=ACTOR_FC1_UNITS, fc2_units=ACTOR_FC2_UNITS):
        """Initialize parameters and build model.
        Params
        ======
            input_dim (int): Input dimension (Dimension of each state)
            output_dim (int): Output dimension (Dimension of each action)
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.nonlin = NON_LIN
        
        # Dense layers
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_dim)
        
        # Normalization layers
        self.bn1 = nn.BatchNorm1d(fc1_units)
        #self.bn2 = nn.BatchNorm1d(fc2_units)
        
        self.reset_parameters()
        

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        
        # Reshape the state to comply with Batch Normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
            
        h1 = self.nonlin(self.fc1(state))
        h1 = self.bn1(h1) # Batch Normalization after Activation  
        h2 = self.nonlin(self.fc2(h1))
        return F.tanh(self.fc3(h2))    



class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_dim, action_size, seed=10, fcs1_units=CRITIC_FCS1_UNITS, fc2_units=CRITIC_FC2_UNITS):
        """Initialize parameters and build model.
        Params
        ======
            input_dim (int): Input dimension (Dimension of each state)
            action_size : Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.nonlin = NON_LIN
        
        # Dense layers 
        
        # Vanilla DDPG architecture
        #self.fcs1 = nn.Linear(input_dim, fcs1_units)
        #self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        
        # Modified DDPG architecture
        self.fcs1 = nn.Linear(input_dim+action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        
        self.fc3 = nn.Linear(fc2_units, 1)
        
        # Normalization layers
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        #self.bn2 = nn.BatchNorm1d(fc2_units)
        
        self.reset_parameters()
        

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
          
        # Reshape the state to comply with Batch Normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)

        # Vanilla DDPG architecture    
        #xs = self.nonlin(self.fcs1(state))
        ###xs = self.bn1(xs) # Batch Normalization after Activation  
        #x = torch.cat((xs, action.float()), dim=1)
        
        # Modified DDPG architecture
        xs = torch.cat((state, action.float()), dim=1)
        x = self.nonlin(self.fcs1(xs))
        x = self.bn1(x) # Batch Normalization after Activation 
        
        x = self.nonlin(self.fc2(x))
        return self.fc3(x)

   