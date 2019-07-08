import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, bn_mode=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            bn_mode (int): Use Batch Normalization - 0=disabled, 1=BN before Activation, 2=BN after Activation (3, 4 are alt. versions of 1, 2)
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Dense layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Normalization layers
        self.bn1 = nn.BatchNorm1d(fc1_units)
        if bn_mode!=2:
            self.bn2 = nn.BatchNorm1d(fc2_units)     
        if bn_mode==3:    
            self.bn3 = nn.BatchNorm1d(action_size)   
        self.bn_mode=bn_mode
        
        self.reset_parameters()
        
        #print("[INFO] Actor initialized with parameters : state_size={} action_size={} seed={} fc1_units={} fc2_units={} bn_mode={}".format(state_size, action_size, seed, fc1_units, fc2_units, bn_mode))
        
        

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
    
        # Reshape the state to comply with Batch Normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        
        if self.bn_mode==0:
            # Batch Normalization disabled
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return F.tanh(self.fc3(x))
        elif self.bn_mode==1:
            # Batch Normalization before Activation
            x = self.fc1(state)
            x = self.bn1(x)   
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)   
            x = F.relu(x)
            x = self.fc3(x)
            return F.tanh(x)
        elif self.bn_mode==2:
            # Batch Normalization after Activation  
            x = F.relu(self.fc1(state))
            x = self.bn1(x) 
            x = F.relu(self.fc2(x))
            return F.tanh(self.fc3(x))
        elif self.bn_mode==3:
            # Batch Normalization before Activation (alternate version)
            x = self.fc1(state)
            x = self.bn1(x)   
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)   
            x = F.relu(x)
            x = self.fc3(x)
            x = self.bn3(x)   
            return F.tanh(x)
        elif self.bn_mode==4:
            # Batch Normalization after Activation  (alternate version)
            x = F.relu(self.fc1(state))
            x = self.bn1(x) 
            x = F.relu(self.fc2(x))
            x = self.bn2(x)   
            return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300, bn_mode=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            bn_mode (int): Use Batch Norm. - 0=disabled, 1=BN before Activation, 2=BN after Activation (3, 4 are alt. versions of 1, 2)
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Dense layers
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        # Normalization layers
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        if bn_mode>2:
            self.bn2 = nn.BatchNorm1d(fc2_units)
        self.bn_mode=bn_mode

        self.reset_parameters()
        
        #print("[INFO] CRITIC initialized with parameters : state_size={} action_size={} seed={} fcs1_units={} fc2_units={} bn_mode={}".format(state_size, action_size, seed, fcs1_units, fc2_units, bn_mode))
        

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        # Reshape the state to comply with Batch Normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
                  
        if self.bn_mode==0:
            # Batch Normalization disabled
            xs = F.relu(self.fcs1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        elif self.bn_mode==1:
            # Batch Normalization before Activation
            xs = self.fcs1(state)
            xs = self.bn1(xs)   
            xs = F.relu(xs)
            x = torch.cat((xs, action), dim=1)
            x = self.fc2(x)
            x = F.relu(x)
            return self.fc3(x)
        elif self.bn_mode==2:
            # Batch Normalization after Activation  
            xs = F.relu(self.fcs1(state))
            xs = self.bn1(xs) 
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        elif self.bn_mode==3:
            # Batch Normalization before Activation (alternate version)
            xs = self.fcs1(state)
            xs = self.bn1(xs)   
            xs = F.relu(xs)
            x = torch.cat((xs, action), dim=1)
            x = self.fc2(x)
            x = self.bn2(x) 
            x = F.relu(x)
            return self.fc3(x)
        elif self.bn_mode==4:
            # Batch Normalization after Activation (alternate version) 
            xs = F.relu(self.fcs1(state))
            xs = self.bn1(xs) 
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            x = self.bn2(x)   
            return self.fc3(x)
            
    