import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        
        fc1_units = 1024 #256
        fc2_units = 1024 #256
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
  
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state))
        #x = F.dropout(x, p=0.0, training=self.training)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=0.0, training=self.training)
        x = self.fc3(x)
        
        return x
        
