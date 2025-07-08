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
    
    CUSTOM Architecture matching the TensorFlow implementation:
    - 1 Hidden Layer (5 neurons, 'elu' activation)
    - Output for mu: Linear
    - Output for sigma: ReLU (to ensure non-negative standard deviation)
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=5):
        """
        Initializes the Actor network.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_action (float): The absolute maximum value of the action.
            hidden_dim (int): Number of neurons in the hidden layer (default: 5).
        """
        super(Actor, self).__init__()
        
        # --- CUSTOM ARCHITECTURE ---
        # 1 hidden layer with 5 neurons and 'elu' activation
        self.hidden1 = nn.Linear(state_dim, hidden_dim)
        
        # Output layer for the mean of the action (linear activation)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        # Output layer for the standard deviation of the action (ReLU activation)
        # Note: We'll add epsilon in forward pass to prevent std from being zero
        self.std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            tuple: A tuple containing:
                - mean (torch.Tensor): The mean of the action distribution.
                - std (torch.Tensor): The standard deviation of the action distribution.
        """
        # Pass through the single hidden layer with ELU activation
        x = F.elu(self.hidden1(state))
        
        # Calculate the mean with linear activation, then apply tanh and scale
        mean = torch.tanh(self.mean_layer(x)) * self.max_action
        
        # Calculate the standard deviation with ReLU activation
        # Add small epsilon to prevent std from being zero
        std = F.relu(self.std_layer(x)) + 1e-6
        
        return mean, std

    def sample(self, state):
        """
        Samples an action from the policy distribution for a given state.
        This also calculates the log probability of the sampled action.
        """
        mean, std = self.forward(state)
        
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
    
    CUSTOM Architecture matching the TensorFlow implementation:
    - 5 Hidden Layers (5 neurons each)
    - Activations: elu -> tanh -> elu -> tanh -> elu
    - Output Layer: Linear
    
    This network takes a state and an action as input and outputs the
    predicted Q-value (the expected return). SAC uses two of these networks.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=5):
        """
        Initializes the Critic network.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of neurons in each hidden layer (default: 5).
        """
        super(Critic, self).__init__()
        
        # --- CUSTOM ARCHITECTURE FOR CRITIC 1 ---
        # 5 hidden layers with specified activations: elu -> tanh -> elu -> tanh -> elu
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)  # elu
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)              # tanh
        self.fc3_q1 = nn.Linear(hidden_dim, hidden_dim)              # elu
        self.fc4_q1 = nn.Linear(hidden_dim, hidden_dim)              # tanh
        self.fc5_q1 = nn.Linear(hidden_dim, hidden_dim)              # elu
        self.out_q1 = nn.Linear(hidden_dim, 1)                       # linear
        
        # --- CUSTOM ARCHITECTURE FOR CRITIC 2 ---
        # 5 hidden layers with specified activations: elu -> tanh -> elu -> tanh -> elu
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)  # elu
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)              # tanh
        self.fc3_q2 = nn.Linear(hidden_dim, hidden_dim)              # elu
        self.fc4_q2 = nn.Linear(hidden_dim, hidden_dim)              # tanh
        self.fc5_q2 = nn.Linear(hidden_dim, hidden_dim)              # elu
        self.out_q2 = nn.Linear(hidden_dim, 1)                       # linear

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
        
        # --- CRITIC 1 Forward Pass ---
        # Pass through all five hidden layers with specified activations
        q1 = F.elu(self.fc1_q1(sa))      # 1st layer: ELU
        q1 = torch.tanh(self.fc2_q1(q1)) # 2nd layer: tanh
        q1 = F.elu(self.fc3_q1(q1))      # 3rd layer: ELU
        q1 = torch.tanh(self.fc4_q1(q1)) # 4th layer: tanh
        q1 = F.elu(self.fc5_q1(q1))      # 5th layer: ELU
        q1 = self.out_q1(q1)             # Output: linear
        
        # --- CRITIC 2 Forward Pass ---
        # Pass through all five hidden layers with specified activations
        q2 = F.elu(self.fc1_q2(sa))      # 1st layer: ELU
        q2 = torch.tanh(self.fc2_q2(q2)) # 2nd layer: tanh
        q2 = F.elu(self.fc3_q2(q2))      # 3rd layer: ELU
        q2 = torch.tanh(self.fc4_q2(q2)) # 4th layer: tanh
        q2 = F.elu(self.fc5_q2(q2))      # 5th layer: ELU
        q2 = self.out_q2(q2)             # Output: linear
        
        return q1, q2