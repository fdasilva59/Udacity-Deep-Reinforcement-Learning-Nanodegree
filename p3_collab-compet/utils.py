import numpy as np
import torch

# Helper functions to concatenate/extract multipe agents states/actions for use with the Replay Buffer memory.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def encode(sa):
    """
    Encode an Environment state or action list of array, which contain multiple agents action/state information, 
    by concatenating their information, thus removing (but not loosing) the agent dimension in the final output. 
    
    The ouput is a list intended to be inserted into a buffer memmory originally not designed to handle multiple 
    agents information, such as in the context of MADDPG)
    
    Params
    ======       
            sa (listr) : List of Environment states or actions array, corresponding to each agent
                
    """
    return np.array(sa).reshape(1,-1).squeeze()



def decode(size, num_agents, id_agent, sa, debug=False):
    """
    Decode a batch of Environment states or actions, which have been previously concatened to store 
    multiple agent information into a buffer memmory originally not designed to handle multiple 
    agents information(such as in the context of MADDPG)
    
    This returns a batch of Environment states or actions (torch.tensor) containing the data 
    of only the agent specified.
    
    Params
    ======
            size (int): size of the action space of state spaec to decode
            num_agents (int) : Number of agent in the environment (and for which info hasbeen concatenetaded)
            id_agent (int): index of the agent whose informationis going to be retrieved
            sa (torch.tensor) : Batch of Environment states or actions, each concatenating the info of several 
                                agents (This is sampled from the buffer memmory in the context of MADDPG)
            debug (boolean) : print debug information
    
    """
    
    list_indices  = torch.tensor([ idx for idx in range(id_agent * size, id_agent * size + size) ]).to(device)    
    out = sa.index_select(1, list_indices)
   
    if (debug):
        print("\nDebug decode:\n size=",size, " num_agents=", num_agents, " id_agent=", id_agent, "\n")
        print("input:\n", sa,"\n output:\n",out,"\n\n\n")
    return  out

