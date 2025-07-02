import os
# Fix for OpenMP library conflict (must be set before importing torch/matplotlib)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
import matplotlib
import glob
from datetime import datetime

# Set matplotlib backend to prevent blocking. 
# 'TkAgg' is a good choice for interactive plots.
matplotlib.use('TkAgg')

# Configure matplotlib for non-blocking plots
plt.ion()  # Turn on interactive mode

# --- Import our custom modules ---
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile
from Reward_Function import RewardCalculator
from NNArchitecture import Actor, Critic

# --- OPTIMIZED Hyperparameters ---
STATE_DIM = 4
ACTION_DIM = 1
MAX_ACTION = 100.0
HIDDEN_DIM = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
BUFFER_SIZE = 50000
BATCH_SIZE = 256
TOTAL_EPISODES = 2000
DT = 0.001  
MAX_STEPS_PER_EPISODE = 5000
TRAIN_FREQUENCY = 4
# --- Logging and Checkpoint Parameters ---
LOG_INTERVAL = 1
CHECKPOINT_INTERVAL = 50
MAX_CHECKPOINTS = 10

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
    def __init__(self, checkpoint_dir="checkpoints", plots_dir="training_plots"):
        self.actor = Actor(STATE_DIM, ACTION_DIM, MAX_ACTION, HIDDEN_DIM)
        self.critic = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        self.critic_target = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        
        self.log_alpha = torch.tensor(np.log(ALPHA), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        self.target_entropy = -torch.prod(torch.Tensor((ACTION_DIM,)).to(self.log_alpha.device)).item()
        
        # Checkpoint and plot management
        self.checkpoint_dir = checkpoint_dir
        self.plots_dir = plots_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Keep track of the current plot figure
        self.current_fig = None

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.actor.sample(state_tensor)
            return action.cpu().numpy()[0]

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

    def save_checkpoint(self, episode, episode_reward, avg_reward):
        """Save a checkpoint of the current model state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"sac_checkpoint_ep{episode}_{timestamp}.pt")
        
        checkpoint = {
            'episode': episode,
            'episode_reward': episode_reward,
            'avg_reward': avg_reward,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'hyperparameters': {
                'STATE_DIM': STATE_DIM,
                'ACTION_DIM': ACTION_DIM,
                'MAX_ACTION': MAX_ACTION,
                'HIDDEN_DIM': HIDDEN_DIM,
                'LEARNING_RATE': LEARNING_RATE,
                'GAMMA': GAMMA,
                'TAU': TAU,
                'ALPHA': ALPHA
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        self._cleanup_old_checkpoints()
        
        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last MAX_CHECKPOINTS."""
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "sac_checkpoint_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) > MAX_CHECKPOINTS:
            checkpoint_files.sort(key=os.path.getmtime)
            files_to_remove = checkpoint_files[:-MAX_CHECKPOINTS]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    print(f"Removed old checkpoint: {os.path.basename(file_path)}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint and restore the model state."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        episode = checkpoint['episode']
        episode_reward = checkpoint['episode_reward']
        avg_reward = checkpoint['avg_reward']
        
        print(f"Checkpoint loaded: Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
        
        return episode, episode_reward, avg_reward

    def get_latest_checkpoint(self):
        """Get the path to the most recent checkpoint."""
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "sac_checkpoint_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None
        
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint

    def plot_training_progress(self, episode, all_episode_rewards, avg_rewards_over_time):
        """Plot training progress and save to file (non-blocking)."""
        try:
            if self.current_fig is not None:
                plt.close(self.current_fig)
            
            self.current_fig = plt.figure(figsize=(15, 10))
            gs = self.current_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            ax1 = self.current_fig.add_subplot(gs[0, :])
            episodes = range(1, len(all_episode_rewards) + 1)
            ax1.plot(episodes, all_episode_rewards, label='Episode Reward', alpha=0.6, color='lightblue')
            ax1.plot(episodes, avg_rewards_over_time, label='Avg. Reward (100-episode rolling)', 
                    color='red', linewidth=2)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title(f'SAC Training Progress - Episode {episode}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = self.current_fig.add_subplot(gs[1, 0])
            recent_start = max(0, len(all_episode_rewards) - 200)
            recent_episodes = episodes[recent_start:]
            recent_rewards = all_episode_rewards[recent_start:]
            recent_avg = avg_rewards_over_time[recent_start:]
            
            ax2.plot(recent_episodes, recent_rewards, label='Episode Reward', alpha=0.6, color='lightgreen')
            ax2.plot(recent_episodes, recent_avg, label='Avg. Reward', color='darkgreen', linewidth=2)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Total Reward')
            ax2.set_title('Recent Performance (Last 200 Episodes)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3 = self.current_fig.add_subplot(gs[1, 1])
            if len(all_episode_rewards) >= 50:
                last_50 = all_episode_rewards[-50:]
                stats_text = f"""Training Statistics (Last 50 Episodes):
                
Mean Reward: {np.mean(last_50):.2f}
Std Reward: {np.std(last_50):.2f}
Min Reward: {np.min(last_50):.2f}
Max Reward: {np.max(last_50):.2f}

Overall Statistics:
Episodes Completed: {episode}
Best Episode Reward: {np.max(all_episode_rewards):.2f}
Current Avg (100-ep): {avg_rewards_over_time[-1]:.2f}"""
            else:
                stats_text = f"""Training Statistics:
                
Episodes Completed: {episode}
Current Reward: {all_episode_rewards[-1]:.2f}
Best Reward So Far: {np.max(all_episode_rewards):.2f}
                
(Need 50+ episodes for full stats)"""
            
            ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax3.axis('off')
            ax3.set_title('Performance Summary')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"training_progress_ep{episode}_{timestamp}.png"
            plot_path = os.path.join(self.plots_dir, plot_filename)
            
            self.current_fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training plot saved: {plot_filename}")
            
            # Non-blocking display with a pause to allow rendering
            self.current_fig.show()
            plt.pause(0.1)  # <-- IMPORTANT: This gives the GUI time to draw the plot
            
            self._cleanup_old_plots()
            
        except Exception as e:
            print(f"Warning: Plotting failed with error: {e}")
            print("Training will continue...")
    
    def _cleanup_old_plots(self):
        """Remove old plot files, keeping only the last 20."""
        plot_pattern = os.path.join(self.plots_dir, "training_progress_*.png")
        plot_files = glob.glob(plot_pattern)
        
        if len(plot_files) > 20:
            plot_files.sort(key=os.path.getmtime)
            files_to_remove = plot_files[:-20]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

if __name__ == '__main__':
    print("=== SAC Active Suspension Training (OPTIMIZED) ===")
    print(f"Episode length: {MAX_STEPS_PER_EPISODE} steps ({MAX_STEPS_PER_EPISODE * DT:.1f}s simulation)")
    print(f"Time step: {DT}s")
    print(f"Training frequency: Every {TRAIN_FREQUENCY} steps")
    print(f"Checkpoints will be saved every {CHECKPOINT_INTERVAL} episodes")
    print(f"Buffer size: {BUFFER_SIZE}")
    print()
    
    model = QuarterCarModel(dt=DT)
    model.m_s, model.m_u, model.k_s, model.k_t, model.c_s = 20.4, 15.9, 8799.0, 90000.0, 100.0
    model._build_state_space_matrices()

    road = SquareWaveProfile(period=2.0, amplitude=0.02)
    reward_calc = RewardCalculator()
    
    agent = SACAgent()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    print(f"Checkpoints directory: {agent.checkpoint_dir}")
    print(f"Plots directory: {agent.plots_dir}")
    print()
    
    all_episode_rewards = []
    last_100_rewards = collections.deque(maxlen=100)
    avg_rewards_over_time = []
    
    start_episode = 0
    latest_checkpoint = agent.get_latest_checkpoint()
    if latest_checkpoint:
        response = input(f"Found checkpoint: {os.path.basename(latest_checkpoint)}. Resume training? (y/n): ")
        if response.lower() == 'y':
            start_episode, _, _ = agent.load_checkpoint(latest_checkpoint)
            print(f"Resuming training from episode {start_episode + 1}")
    
    episode_duration = MAX_STEPS_PER_EPISODE * DT
    time_vector = np.arange(0, episode_duration, DT)
    road_profile = road.get_profile(time_vector)
    print(f"Pre-generated road profile for {episode_duration}s episodes")
    
    training_start_time = datetime.now()
    
    for episode in range(start_episode, TOTAL_EPISODES):
        state = model.reset()
        episode_reward = 0
        
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
            
            if len(replay_buffer) >= BATCH_SIZE and step % TRAIN_FREQUENCY == 0:
                agent.train(replay_buffer)

        all_episode_rewards.append(episode_reward)
        last_100_rewards.append(episode_reward)
        avg_reward = np.mean(last_100_rewards)
        avg_rewards_over_time.append(avg_reward)
        
        episodes_completed = episode - start_episode + 1
        elapsed_time = (datetime.now() - training_start_time).total_seconds()
        avg_time_per_episode = elapsed_time / episodes_completed if episodes_completed > 0 else 0
        estimated_total_time = avg_time_per_episode * (TOTAL_EPISODES - start_episode)
        estimated_remaining = estimated_total_time - elapsed_time
        
        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"Ep: {episode+1}/{TOTAL_EPISODES} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg(100): {avg_reward:.1f} | "
                  f"Time: {avg_time_per_episode:.1f}s/ep | "
                  f"ETA: {estimated_remaining/60:.1f}min")
        
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            agent.save_checkpoint(episode + 1, episode_reward, avg_reward)
            agent.plot_training_progress(episode + 1, all_episode_rewards, avg_rewards_over_time)

    print("Training completed. Saving final checkpoint...")
    agent.save_checkpoint(TOTAL_EPISODES, all_episode_rewards[-1], avg_rewards_over_time[-1])
    
    print("Generating final training progress plot...")
    agent.plot_training_progress(TOTAL_EPISODES, all_episode_rewards, avg_rewards_over_time)
    
    total_training_time = (datetime.now() - training_start_time).total_seconds()
    print(f"Training completed in {total_training_time/3600:.2f} hours!")
    print("Check the plots directory for saved training progress plots.")

    # Keep the final plot window open until you manually close it
    print("Close the final plot window to exit the script.")
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the plot and block until it's closed
