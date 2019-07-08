import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed=10,
                actor_fc1_units=400, actor_fc2_units=300,
                critic_fcs1_units=400, critic_fc2_units=300,
                buffer_size=int(1e5), batch_size=128,
                gamma=0.99, tau=1e-3 , bn_mode=0,
                lr_actor=1e-4, lr_critic=1e-4, weight_decay=0,
                add_ounoise=True, mu=0., theta=0.15, sigma=0.2):
        
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            actor_fc1_units (int): number of units for the layer 1 in the actor model
            actor_fc2_units (int): number of units for the layer 2 in the actor model
            critic_fcs1_units (int): number of units for the layer 1 in the critic model
            critic_fc2_units (int): number of units for the layer 2 in the critic model
            buffer_size (int) : replay buffer size
            batch_size (int) : minibatch size
            gamma (int) : discount factor
            tau (int) : for soft update of target parameter
            bn_mode (int): Use Batch Norm - 0=disabled, 1=BN before Activation, 2=BN after Activation (3, 4 are alt. versions of 1, 2)
            lr_actor (int) : learning rate of the actor 
            lr_critic (int) : learning rate of the critic 
            weight_decay (int) : L2 weight decay  
            add_ounoise (bool) : Add Ornstein-Uhlenbeck noise
            mu (float) : Ornstein-Uhlenbeck noise parameter
            theta (float) : Ornstein-Uhlenbeck noise parameter
            sigma (float) : Ornstein-Uhlenbeck noise parameter            
        """
        
        print("\n[INFO]ddpg constructor called with parameters: state_size={} action_size={} random_seed={} actor_fc1_units={} actor_fc2_units={} critic_fcs1_units={} critic_fc2_units={} buffer_size={} batch_size={} gamma={} tau={} bn_mode={} lr_actor={} lr_critic={} weight_decay={} add_ounoise={} mu={} theta={} sigma={} \n".format(state_size, action_size, random_seed, 
                                                          actor_fc1_units, actor_fc2_units, critic_fcs1_units, critic_fc2_units,
                                                          buffer_size, batch_size, gamma, tau, 
                                                          bn_mode, lr_actor, lr_critic, weight_decay, 
                                                          add_ounoise, mu, theta,sigma))
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.buffer_size=buffer_size
        self.batch_size=batch_size
        self.gamma=gamma
        self.tau=tau
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.weight_decay=weight_decay
        self.add_ounoise=add_ounoise

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, actor_fc1_units, actor_fc2_units, bn_mode=bn_mode).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, actor_fc1_units, actor_fc2_units, bn_mode=bn_mode).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Make sure the Target Network has the same weight values as the Local Network
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)
    
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, critic_fcs1_units, critic_fc2_units, bn_mode=bn_mode).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, critic_fcs1_units, critic_fc2_units, bn_mode=bn_mode).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=self.weight_decay)
        
        # Make sure the Target Network has the same weight values as the Local Network
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

        # Noise process
        self.noise = OUNoise(size=action_size, seed=random_seed, 
                             mu=mu, theta=theta, sigma=sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.add_ounoise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss (using gradient clipping)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)