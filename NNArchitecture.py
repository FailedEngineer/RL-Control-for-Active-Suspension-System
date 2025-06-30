import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# A small constant for numerical stability
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    """
    The Actor (Policy) network for the SAC algorithm.

    This network takes the current state of the environment as input and outputs
    the parameters of a probability distribution (mean and standard deviation)
    from which the action is sampled.
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        """
        Initializes the Actor network.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_action (float): The absolute maximum value of the action.
            hidden_dim (int): Number of neurons in the hidden layers.
        """
        super(Actor, self).__init__()
        
        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer for the mean of the action
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        # Output layer for the log standard deviation of the action
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            tuple: A tuple containing:
                - mean (torch.Tensor): The mean of the action distribution.
                - log_std (torch.Tensor): The log standard deviation of the action distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Calculate the mean, using tanh to constrain it between -1 and 1
        mean = torch.tanh(self.mean_layer(x))
        
        # Calculate the log standard deviation
        log_std = self.log_std_layer(x)
        # Clamp the log_std for numerical stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std

    def sample(self, state):
        """
        Samples an action from the policy distribution for a given state.
        This also calculates the log probability of the sampled action.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create a Normal (Gaussian) distribution
        normal = Normal(mean, std)
        
        # Sample an action (reparameterization trick for differentiability)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Scale the action to the actuator's limits
        scaled_action = action * self.max_action
        
        # Calculate the log probability of the action
        log_prob = normal.log_prob(x_t)
        # Enforce the action bounds (correction due to tanh)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return scaled_action, log_prob


class Critic(nn.Module):
    """
    The Critic (Q-Value) network for the SAC algorithm.

    This network takes a state and an action as input and outputs the
    predicted Q-value (the expected return). SAC uses two of these networks.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initializes the Critic network.
        """
        super(Critic, self).__init__()
        
        # --- Critic 1 Layers ---
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.out_q1 = nn.Linear(hidden_dim, 1) # Output is a single Q-value
        
        # --- Critic 2 Layers ---
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_q2 = nn.Linear(hidden_dim, 1) # Output is a single Q-value

    def forward(self, state, action):
        """
        Performs the forward pass for both critics.
        
        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.
            
        Returns:
            tuple: A tuple containing the Q-values from both critics.
        """
        # Concatenate state and action for the input
        sa = torch.cat([state, action], 1)
        
        # --- Critic 1 Forward Pass ---
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.out_q1(q1) # Linear output
        
        # --- Critic 2 Forward Pass ---
        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.out_q2(q2) # Linear output
        
        return q1, q2
