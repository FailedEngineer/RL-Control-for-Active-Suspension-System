import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import random
import matplotlib.pyplot as plt

# --- Import our custom modules ---
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile
from Reward_Function import RewardCalculator
from NNArchitecture import Actor, Critic

# --- Hyperparameters ---
STATE_DIM = 4
ACTION_DIM = 1
MAX_ACTION = 720.0
HIDDEN_DIM = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
BUFFER_SIZE = 1000000
BATCH_SIZE = 256
TOTAL_EPISODES = 2000
DT = 0.001
MAX_STEPS_PER_EPISODE = 5000
# --- NEW: Hyperparameter for logging frequency ---
LOG_INTERVAL = 10 # Print log every 10 episodes

class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    """The Soft Actor-Critic Agent."""
    def __init__(self):
        self.actor = Actor(STATE_DIM, ACTION_DIM, MAX_ACTION, HIDDEN_DIM)
        self.critic = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        self.critic_target = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        
        self.log_alpha = torch.tensor(np.log(ALPHA), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        self.target_entropy = -torch.prod(torch.Tensor((ACTION_DIM,)).to(self.log_alpha.device)).item()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor.sample(state_tensor)
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_target, q2_target = self.critic_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target)
            value_target = reward + (1 - done) * GAMMA * (q_target - self.log_alpha.exp() * next_log_prob)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, value_target) + F.mse_loss(q2, value_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


if __name__ == '__main__':
    model = QuarterCarModel(dt=DT)
    model.m_s, model.m_u, model.k_s, model.k_t, model.c_s = 20.4, 15.9, 8799.0, 90000.0, 100.0
    model._build_state_space_matrices()

    road = SquareWaveProfile(period=2.0, amplitude=0.02)
    reward_calc = RewardCalculator()
    
    agent = SACAgent()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    # --- NEW: Data structures for improved logging ---
    all_episode_rewards = []
    # Use a deque for an efficient rolling window of the last 100 rewards
    last_100_rewards = collections.deque(maxlen=100)
    avg_rewards_over_time = []
    
    for episode in range(TOTAL_EPISODES):
        state = model.reset()
        episode_reward = 0
        
        time_vector = np.arange(0, MAX_STEPS_PER_EPISODE * DT, DT)
        road_profile = road.get_profile(time_vector)

        for step in range(MAX_STEPS_PER_EPISODE):
            if len(replay_buffer) < BATCH_SIZE:
                action = np.random.uniform(-MAX_ACTION, MAX_ACTION, size=(ACTION_DIM,))
            else:
                action = agent.select_action(state)
            
            current_road_height = road_profile[step]
            
            next_state, x_s_ddot, p_regen = model.step(action[0], current_road_height)
            
            reward = reward_calc.calculate_reward(p_regen, x_s_ddot, state[2], current_road_height, action[0])
            
            done = (step == MAX_STEPS_PER_EPISODE - 1)
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) >= BATCH_SIZE:
                agent.train(replay_buffer)

        all_episode_rewards.append(episode_reward)
        last_100_rewards.append(episode_reward)
        avg_reward = np.mean(last_100_rewards)
        avg_rewards_over_time.append(avg_reward)
        
        # --- NEW: Print log at specified intervals ---
        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"Episode: {episode+1}/{TOTAL_EPISODES} | Reward: {episode_reward:.2f} | Avg. Reward (Last 100): {avg_reward:.2f}")

    # --- NEW: More informative plotting at the end of training ---
    plt.figure(figsize=(12, 6))
    plt.plot(all_episode_rewards, label='Episode Reward', alpha=0.5)
    plt.plot(avg_rewards_over_time, label='Avg. Reward (100-episode rolling)', color='red', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SAC Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()
