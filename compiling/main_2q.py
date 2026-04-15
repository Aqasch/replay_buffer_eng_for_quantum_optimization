"""
Owned by Akash (https://aqasch.github.io) 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import numpy as np
import random
import pandas as pd
import os
import time
import argparse
import pickle

from environment_compiling_n_qubit_haar_random import CircuitEnv

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


# ========== Standard DQN Network (from paper Table 4) ==========

class StandardDQN(nn.Module):
    """Standard DQN with SELU activations (from paper Table 4)"""
    def __init__(self, state_size, action_size):
        super().__init__()
        
        # Table 4: 128, 128 hidden layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """LeCun initialization (Table 4: initializers = lecun, lecun, glorot)"""
        nn.init.normal_(self.fc1.weight, mean=0, std=np.sqrt(1/self.fc1.in_features))
        nn.init.normal_(self.fc2.weight, mean=0, std=np.sqrt(1/self.fc2.in_features))
        nn.init.xavier_uniform_(self.fc3.weight)  # Glorot = Xavier
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        """Forward pass with SELU activations (Table 4)"""
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)  # Linear output
        return x


# ========== Prioritized Experience Replay ==========

class SumTree:
    """
    Binary tree for efficient sampling in PER
    Leaf nodes store priorities, internal nodes store sums
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Update to the root node"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        """Store priority and sample"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update priority"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get priority and sample"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER)
    Based on: Schaul et al. (2016) "Prioritized Experience Replay"
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, device='cpu'):
        """
        Args:
            capacity: Buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
            device: torch device
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.device = device
        self.epsilon = 1e-6  # Small constant to avoid zero priority
    
    def _get_beta(self):
        """Linearly anneal beta from beta_start to 1.0"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority"""
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        
        experience = Experience(state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)
    
    def sample(self, batch_size):
        """Sample batch with priorities"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        beta = self._get_beta()
        self.frame += 1
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        
        # Compute importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        
        # Convert to tensors
        states = torch.stack([e.state for e in batch])
        actions = torch.tensor([e.action for e in batch], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32)
        is_weights = torch.tensor(is_weights, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones, idxs, is_weights
    
    def update_priorities(self, idxs, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries

# ========== Prioritized Experience Replay ==========

# ========== Reliability Adjusted Prioritized Experience Replay ==========
class ReliabilityAdjustedPrioritizedReplayBuffer:
    """
    ReaPER: Reliability-Adjusted Prioritized Experience Replay

    Optimized version with per-episode bookkeeping for speed.
    NOW WITH CONFIGURABLE OMEGA ANNEALING!
    """

    def __init__(self, capacity, alpha=0.4, omega=0.2, 
                 omega_anneal=False, omega_start=0.6, omega_end=0.2, omega_frames=50000,
                 beta_start=0.4, beta_frames=100000, device='cpu'):
        """
        Args:
            capacity: Buffer size
            alpha: TD error exponent (ReaPER uses 0.4 vs PER's 0.6)
            omega: Reliability exponent (used if omega_anneal=False)
            omega_anneal: If True, anneal omega from omega_start to omega_end
            omega_start: Initial omega value (e.g., 0.6 for strong reliability)
            omega_end: Final omega value (e.g., 0.2 for weak reliability)
            omega_frames: Number of frames to anneal omega over
            beta_start: Initial importance sampling weight
            beta_frames: Frames to anneal beta to 1.0
            device: torch device
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        
        # NEW: Omega annealing configuration
        self.omega_anneal = omega_anneal
        if omega_anneal:
            self.omega_start = omega_start
            self.omega_end = omega_end
            self.omega_frames = omega_frames
            self.omega_fixed = None  # Not used when annealing
        else:
            self.omega_fixed = omega
            self.omega_start = None
            self.omega_end = None
            self.omega_frames = None
        
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.device = device
        self.epsilon = 1e-6

        # Episode tracking structures
        self.episode_ids = np.zeros(capacity, dtype=np.int32)
        self.current_episode_id = 0
        
        # Store individual TD errors for reliability calculation
        self.td_errors = np.zeros(capacity, dtype=np.float32)

        # Per-episode bookkeeping for fast lookups
        self.episode_indices = {}      # episode_id -> list of buffer indices in temporal order
        self.episode_tde_sums = {}     # episode_id -> cached sum of |TD errors|
        self.episode_complete = {}     # episode_id -> bool (is episode finished?)

        print(f"[ReaPER Buffer] Omega annealing: {omega_anneal}")
        if omega_anneal:
            print(f"  Omega schedule: {omega_start} → {omega_end} over {omega_frames} frames")
        else:
            print(f"  Fixed omega: {omega}")

    def _get_beta(self):
        """Linearly anneal beta from beta_start to 1.0"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def _get_omega(self):
        """
        Get current omega value.
        """
        if self.omega_anneal:
            progress = min(1.0, self.frame / self.omega_frames)
            return self.omega_start + progress * (self.omega_end - self.omega_start)
        else:
            return self.omega_fixed

    def get_current_omega(self):
        return self._get_omega()

    def push(self, state, action, reward, next_state, done):
        """
        Add transition with maximum priority and track episode membership.
        """
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0

        experience = Experience(state, action, reward, next_state, done)

        # Track episode ID for this transition
        buffer_idx = self.tree.write
        ep_id = self.current_episode_id
        self.episode_ids[buffer_idx] = ep_id
        self.td_errors[buffer_idx] = 0.0  # Initialize TD error

        # Register this buffer index in the episode's index list
        if ep_id not in self.episode_indices:
            self.episode_indices[ep_id] = []
            self.episode_tde_sums[ep_id] = 0.0
            self.episode_complete[ep_id] = False
        
        self.episode_indices[ep_id].append(buffer_idx)

        # Add to sum tree
        self.tree.add(max_priority, experience)

        # Mark episode as complete and advance counter
        if done:
            self.episode_complete[ep_id] = True
            self.current_episode_id += 1

    def _compute_reliability(self, buffer_idx):
        """
        Compute reliability score R_t for a given buffer index.
        """
        if self.tree.n_entries == 0:
            return 1.0

        ep_id = int(self.episode_ids[buffer_idx])
        
        # Get cached episode indices (already in temporal order)
        indices = self.episode_indices.get(ep_id, [])
        if not indices:
            return 1.0

        # Get cached total episode TDE sum
        total_tde = self.episode_tde_sums.get(ep_id, 0.0)
        if total_tde < self.epsilon:
            return 1.0

        # Find position of current transition in the episode
        try:
            pos = indices.index(buffer_idx)
        except ValueError:
            return 1.0

        # Compute downstream TDE sum (all transitions after current one)
        downstream_tde = 0.0
        for j in range(pos + 1, len(indices)):
            idx_j = indices[j]
            downstream_tde += abs(self.td_errors[idx_j])

        # Reliability formula from paper
        reliability = 1.0 - (downstream_tde / total_tde)
        
        return max(0.0, min(1.0, reliability))

    def sample(self, batch_size):
        """
        Sample batch with ReaPER priorities.
        """
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        beta = self._get_beta()
        self.frame += 1

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Compute importance sampling weights (same as PER)
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        # Convert to tensors
        states = torch.stack([e.state for e in batch])
        actions = torch.tensor([e.action for e in batch], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32)
        is_weights = torch.tensor(is_weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        """
        Update priorities using reliability-adjusted formula.
        """
        omega = self._get_omega()  # Get current omega (annealed or fixed)
        
        for idx, td_error in zip(idxs, td_errors):
            buffer_idx = idx - self.tree.capacity + 1

            ep_id = int(self.episode_ids[buffer_idx])
            
            # Update cached episode TDE sum incrementally
            old_abs = abs(self.td_errors[buffer_idx])
            new_abs = abs(td_error)
            
            # Store new TD error
            self.td_errors[buffer_idx] = td_error
            
            # Update cached sum: remove old, add new
            if ep_id in self.episode_tde_sums:
                self.episode_tde_sums[ep_id] += (new_abs - old_abs)

            # Compute reliability using cached data
            reliability = self._compute_reliability(buffer_idx)

            # Reliability-adjusted priority (Equation 5 from paper)
            # NOW WITH DYNAMIC OMEGA!
            abs_td_error = new_abs + self.epsilon
            adjusted_priority = (reliability ** omega) * (abs_td_error ** self.alpha)

            # Update priority in sum tree
            self.tree.update(idx, adjusted_priority)

    def __len__(self):
        return self.tree.n_entries

# ============== Reliability Adjusted Prioritized Experience Replay ==========



# ========== Hindsight Experience Replay Buffer ==========

class HindsightReplayBuffer:
    """
    Replay buffer with Hindsight Experience Replay (HER)
    """
    
    def __init__(self, capacity, her_k=4, her_strategy='future', device='cpu'):
        self.buffer = deque(maxlen=capacity)
        self.her_k = her_k  # Number of HER replays per transition
        self.her_strategy = her_strategy
        self.current_episode = []
        self.device = device
    
    def push_transition(self, state, action, reward, next_state, done, 
                       achieved_unitary, target_unitary, env):
        """
        Store transition with goal information for HER
        
        Args:
            state: Current state
            action: Action taken
            reward: Original reward received
            next_state: Next state
            done: Episode done flag
            achieved_unitary: The unitary we actually built (U_n)
            target_unitary: The unitary we wanted to build (U_target)
            env: Environment (needed to recompute rewards for HER)
        """
        self.current_episode.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'achieved_unitary': achieved_unitary.copy(),  # Copy to avoid reference issues
            'target_unitary': target_unitary.copy(),
        })
        
        if done:
            self._store_episode_with_her(env)
            self.current_episode = []
    
    def _store_episode_with_her(self, env):
        """
        Store episode with HER modifications.
        """
        episode_length = len(self.current_episode)
        
        for t, transition in enumerate(self.current_episode):
            # 1. Store original transition (with original goal)
            self.buffer.append(Experience(
                state=transition['state'],
                action=transition['action'],
                reward=transition['reward'],
                next_state=transition['next_state'],
                done=transition['done']
            ))
            
            # 2. Generate HER experiences using 'future' strategy
            # "We designed a strategy to select the new goals, consisting in 
            # randomly selecting k-percent of the states that come from the same episode."
            if self.her_strategy == 'future' and t < episode_length - 1:
                # Sample k future achieved unitaries from same episode
                future_indices = list(range(t + 1, episode_length))
                num_samples = min(self.her_k, len(future_indices))
                
                if num_samples > 0:
                    sampled_indices = random.sample(future_indices, num_samples)
                    
                    for future_t in sampled_indices:
                        # Use future achieved unitary as the "goal"
                        future_achieved = self.current_episode[future_t]['achieved_unitary']
                        
                        # Recompute reward with new goal
                        her_reward = self._compute_reward_with_new_goal(
                            transition['achieved_unitary'],
                            future_achieved,
                            env
                        )
                        
                        # Check if we "solved" the new goal
                        her_done = self._check_success(
                            transition['achieved_unitary'],
                            future_achieved,
                            env.tolerance
                        )
                        
                        # Store HER experience
                        self.buffer.append(Experience(
                            state=transition['state'],
                            action=transition['action'],
                            reward=her_reward,
                            next_state=transition['next_state'],
                            done=her_done
                        ))
    
    def _compute_reward_with_new_goal(self, achieved_unitary, goal_unitary, env):
        """
        Compute sparse reward (Equation 4 from paper) with new goal
        r = 0 if fidelity >= tolerance, else -1/L
        """
        fidelity = self._compute_fidelity(achieved_unitary, goal_unitary)
        
        # Sparse reward (Equation 4)
        if fidelity >= env.tolerance:
            return 0.0
        else:
            return -1.0 / env.max_episode_length
    
    def _check_success(self, achieved_unitary, goal_unitary, tolerance):
        """Check if achieved goal matches desired goal within tolerance"""
        fidelity = self._compute_fidelity(achieved_unitary, goal_unitary)
        return fidelity >= tolerance
    
    def _compute_fidelity(self, U1, U2):
        """
        Compute average gate fidelity between two unitaries
        AGF = (|Tr(U1^† @ U2)|^2 + d) / (d(d+1))
        """
        d = U1.shape[0]
        trace = np.trace(U1.conj().T @ U2)
        fidelity = (np.abs(trace) ** 2 + d) / (d * (d + 1))
        return float(fidelity)
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.stack([e.state for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# ========== DQN Agent with PER ==========

class DQNAgentWithPER:
    """
    DQN Agent with Prioritized Experience Replay
    """
    
    def __init__(self, state_size, action_size, config, device, network_type='paper', seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.network_type = network_type
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.99931)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 200)
        self.target_update_freq = config.get('target_update_freq', 1)
        
        # PER configuration
        self.per_alpha = config.get('per_alpha', 0.6)
        self.per_beta_start = config.get('per_beta_start', 0.4)
        self.per_beta_frames = config.get('per_beta_frames', 100000)
        
        # Network selection (same as before)
        if network_type == 'paper':
            self.policy_net = StandardDQN(state_size, action_size).to(device)
            self.target_net = StandardDQN(state_size, action_size).to(device)
        else:
            raise ValueError(f"Unknown network_type: {network_type}")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # PER buffer (replaces HER buffer)
        memory_size = config.get('memory_size', 500000)
        self.memory = PrioritizedReplayBuffer(
            capacity=memory_size,
            alpha=self.per_alpha,
            beta_start=self.per_beta_start,
            beta_frames=self.per_beta_frames,
            device=device
        )
        
        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.losses = []
        
        print(f"DQN+PER:")
        print(f"  Network type: {network_type}")
        print(f"  Policy net params: {sum(p.numel() for p in self.policy_net.parameters())}")
        print(f"  Memory size: {memory_size}")
        print(f"  PER alpha: {self.per_alpha}")
        print(f"  PER beta: {self.per_beta_start} → 1.0")
        print(f"  Batch size: {self.batch_size}")
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state.unsqueeze(0))
                return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """storage"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """DQN training step with PER"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from PER buffer (with priorities)
        states, actions, rewards, next_states, dones, idxs, is_weights = \
            self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD errors for priority update
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        
        # Weighted loss (importance sampling)
        loss = (is_weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities in buffer
        self.memory.update_priorities(idxs, td_errors)
        
        self.steps_done += 1
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'network_type': self.network_type,
        }, filepath)
        print(f"DQN+PER model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        print(f"DQN+PER model loaded from {filepath}")

# ========== DQN Agent with PER ==========

class DQNAgentWithReaPER:
    """
    DQN Agent with Reliability-Adjusted Prioritized Experience Replay (Pleiss, Leonard S. ICLR 2026.)
    """

    def __init__(self, state_size, action_size, config, device, network_type='paper', seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.network_type = network_type

        # Hyperparameters (same as PER)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.99931)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 200)
        self.target_update_freq = config.get('target_update_freq', 1)

        # ReaPER configuration (different from PER!)
        self.reaper_alpha = config.get('reaper_alpha', 0.4)      # Lower than PER's 0.6
        self.reaper_omega = config.get('reaper_omega', 0.2)      # NEW: Reliability weight
        self.reaper_beta_start = config.get('reaper_beta_start', 0.4)
        self.reaper_beta_frames = config.get('reaper_beta_frames', 100000)

        self.anneal_active = config.get('reaper_omega_anneal', False)

        # Network selection (identical to PER)
        if network_type == 'paper':
            self.policy_net = StandardDQN(state_size, action_size).to(device)
            self.target_net = StandardDQN(state_size, action_size).to(device)
        else:
            raise ValueError(f"Unknown network_type: {network_type}")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer (same as PER)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        memory_size = config.get('memory_size', 500000)
        self.memory = ReliabilityAdjustedPrioritizedReplayBuffer(
                capacity=memory_size,
                alpha=config.get('reaper_alpha', 0.4),
                omega_anneal=config.get('reaper_omega_anneal', False),
                omega_start=config.get('reaper_omega_start', 0.7),
                omega_end=config.get('reaper_omega_end', 0.15),
                omega_frames=config.get('reaper_omega_frames', 50000),
                beta_start=config.get('reaper_beta_start', 0.4),
                beta_frames=config.get('reaper_beta_frames', 100000),
                device=device
            )

        # Tracking (same as PER)
        self.steps_done = 0
        self.episodes_done = 0
        self.losses = []

        print(f"DQN+ReaPER Agent initialized:")
        print(f"  Network type: {network_type}")
        print(f"  Policy net params: {sum(p.numel() for p in self.policy_net.parameters())}")
        print(f"  Memory size: {memory_size}")
        print(f"  ReaPER alpha: {self.reaper_alpha} (vs PER's 0.6)")
        print(f"  ReaPER omega: {self.reaper_omega} (reliability weight)")
        print(f"  ReaPER beta: {self.reaper_beta_start} → 1.0")
        print(f"  Batch size: {self.batch_size}")

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection (identical to PER)"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state.unsqueeze(0))
                return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Simple storage (identical to PER)"""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        DQN training step with ReaPER
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from ReaPER buffer (with reliability-adjusted priorities)
        states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute TD errors for priority update
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()

        # Weighted loss (importance sampling)
        loss = (is_weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # ⭐ Update priorities in buffer (ReaPER does the reliability magic here!)
        self.memory.update_priorities(idxs, td_errors)

        self.steps_done += 1

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.losses.append(loss.item())
        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon (identical to PER)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model checkpoint (identical to PER)"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'network_type': self.network_type,
        }, filepath)
        print(f"DQN+ReaPER model saved to {filepath}")

    def load(self, filepath):
        """Load model checkpoint (identical to PER)"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        print(f"DQN+ReaPER model loaded from {filepath}")


# ========== DQN Agent with ReaPER ==========


# ========== DQN Agent with HER ==========

class DQNAgentWithHER:
    """
    DQN Agent with Hindsight Experience Replay for HRC gates
    """
    
    def __init__(self, state_size, action_size, config, device, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters from Table 4
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.99931)  # Table 4
        self.learning_rate = config.get('learning_rate', 1e-4)  # Table 4: 0.0001
        self.batch_size = config.get('batch_size', 200)  # Table 4
        self.target_update_freq = config.get('target_update_freq', 1)  # Every episode
        
        # HER configuration
        self.her_k = config.get('her_k', 4)
        self.her_strategy = config.get('her_strategy', 'future')
        
        # Networks (Table 4: SELU activations, LeCun/Glorot initialization)
        self.policy_net = StandardDQN(state_size, action_size).to(device)
        self.target_net = StandardDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer (Table 4: Adam)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # HER replay buffer (Table 4: memory size = 5·10^5)
        memory_size = config.get('memory_size', 500000)
        self.memory = HindsightReplayBuffer(
            capacity=memory_size,
            her_k=self.her_k,
            her_strategy=self.her_strategy,
            device=device
        )
        
        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.losses = []
        
        print(f"DQN+HER Agent initialized:")
        print(f"  Policy net params: {sum(p.numel() for p in self.policy_net.parameters())}")
        print(f"  Memory size: {memory_size}")
        print(f"  HER k: {self.her_k}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epsilon decay: {self.epsilon_decay}")
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state.unsqueeze(0))
                return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done,
                        achieved_unitary, target_unitary, env):
        """Store transition with goal information for HER"""
        self.memory.push_transition(
            state, action, reward, next_state, done,
            achieved_unitary, target_unitary, env
        )
    
    def train_step(self):
        """
        DQN training step
        Table 4: training frequency = every one episode
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from HER buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: select actions with policy net, evaluate with target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        
        # Update target network (Table 4: every 1 episode in practice)
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon (Table 4: epsilon decay = 0.99931)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, filepath)
        print(f"DQN+HER model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        print(f"DQN+HER model loaded from {filepath}")


# ========== Evaluation Function ==========

def evaluate_agent(env, agent, num_episodes=100):
    """
    Evaluate agent performance WITHOUT training
    Returns success rate and average fidelity
    """
    agent.policy_net.eval()  # Set to evaluation mode
    
    successes = 0
    fidelities = []
    sequence_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Greedy action selection (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action, train_flag=False)
            state = next_state
        
        fidelity = env.average_gate_fidelity()
        fidelities.append(fidelity)
        
        info = env.get_episode_info()
        sequence_lengths.append(info['steps'])
        
        if info['solved']:
            successes += 1
    
    agent.policy_net.train()  # Set back to training mode
    
    return {
        'success_rate': successes / num_episodes,
        'mean_fidelity': np.mean(fidelities),
        'mean_length': np.mean(sequence_lengths),
        'std_length': np.std(sequence_lengths),
    }


# ========== Training Function with Evaluation ==========

def train_dqn(env, agent, replay, anneal_active, config, num_qubits, seed=42):
    """
    Training loop for DQN+HER agent with HRC gates
    """
    num_episodes = config.get('num_episodes', 100000)
    max_steps_per_episode = config.get('max_steps_per_episode', 130)
    print_every = config.get('print_every', 1000)
    save_every = config.get('save_every', 10000)
    eval_every = config.get('eval_every', 1000)  # NEW: Evaluate every N episodes
    eval_episodes = config.get('eval_episodes', 100)  # NEW: Number of eval episodes

    updates_per_episode = config.get('updates_per_episode', 50)
    warmup_episodes = config.get('warmup_episodes', 100)
    
    episode_data = {}
    episode_rewards = []
    episode_lengths = []
    episode_fidelities = []
    success_rate_window = deque(maxlen=100)
    
    # NEW: Track evaluation metrics
    eval_results = []
    
    global_step = 0
    time_init = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        target_unitary = env.target_unitary.copy()  # Save target for HER
        episode_reward = 0
        done = False
        step_in_episode = 0
        
        episode_num = episode + 1
        episode_data[episode_num] = {
            'errors': [],
            'losses': [],
            'fidelities': [],
            'actions': [],
            'rewards': [],
            'times': []
        }
        
        # Collect episode
        while not done:
            # Select action with epsilon-greedy
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action, train_flag=True)
            
            # Get achieved unitary (what we actually built)
            achieved_unitary = env.current_unitary.copy()
            
            if replay == 'her':
                # Store transition with goal information for HER
                agent.store_transition(
                    state, action, reward.item(), next_state, done,
                    achieved_unitary, target_unitary, env
                )
            elif replay == 'per':
                # Store transition normally for PER
                agent.store_transition(
                    state, action, reward.item(), next_state, done
                )
                
            elif replay == 'reaper':
                # Store transition normally for ReaPER
                agent.store_transition(
                    state, action, reward.item(), next_state, done
                )
            elif replay == 'reaper_anneal':
                # Store transition normally for ReaPER
                agent.store_transition(
                    state, action, reward.item(), next_state, done
                )


            current_fidelity = env.average_gate_fidelity()
            current_error = 1.0 - current_fidelity
            
            episode_data[episode_num]['errors'].append(current_error)
            episode_data[episode_num]['fidelities'].append(current_fidelity)
            episode_data[episode_num]['actions'].append(action)
            episode_data[episode_num]['rewards'].append(reward.item())
            episode_data[episode_num]['times'].append(time.time() - time_init)
            
            state = next_state
            episode_reward += reward.item()
            step_in_episode += 1
            global_step += 1
        
        if episode >= warmup_episodes:
            for _ in range(updates_per_episode):
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.train_step()
                    if loss is not None:
                        episode_data[episode_num]['losses'].append(loss)
        
        # Train after episode (Table 4: training frequency = every one episode)
        # loss = agent.train_step()
        # if loss is not None:
            # episode_data[episode_num]['losses'].append(loss)
        
        # Decay epsilon (Table 4: epsilon decay = 0.99931)
        agent.decay_epsilon()
        agent.episodes_done += 1
        
        # Track metrics
        info = env.get_episode_info()
        episode_rewards.append(episode_reward)
        episode_lengths.append(info['steps'])
        episode_fidelities.append(info['fidelity'])
        success_rate_window.append(1.0 if info['solved'] else 0.0)
        
        episode_data[episode_num]['episode_summary'] = {
            'total_reward': episode_reward,
            'episode_length': info['steps'],
            'final_fidelity': info['fidelity'],
            'solved': info['solved'],
            'epsilon': agent.epsilon
        }
        
        # NEW: Periodic evaluation
        if (episode + 1) % eval_every == 0:
            print(f"\n{'='*60}")
            print(f"EVALUATION at Episode {episode + 1}")
            print(f"{'='*60}")
            
            eval_result = evaluate_agent(env, agent, num_episodes=eval_episodes)
            eval_result['episode'] = episode + 1
            eval_results.append(eval_result)
            
            print(f"  Eval Success Rate: {eval_result['success_rate']:.2%}")
            print(f"  Eval Mean Fidelity: {eval_result['mean_fidelity']:.6f}")
            print(f"  Eval Mean Length: {eval_result['mean_length']:.1f} ± {eval_result['std_length']:.1f}")
            print(f"{'='*60}\n")
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            avg_fidelity = np.mean(episode_fidelities[-print_every:])
            success_rate = np.mean(success_rate_window)
            avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Training Avg Reward: {avg_reward:.4f}")
            print(f"  Training Avg Length: {avg_length:.1f}")
            print(f"  Training Avg Fidelity: {avg_fidelity:.6f}")
            print(f"  Training Success Rate (last 100): {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Avg Loss: {avg_loss:.6f}")
            print(f"  Memory size: {len(agent.memory)}")
            print(f"  Total Steps: {global_step}")

            if hasattr(agent, "memory") and hasattr(agent.memory, "get_current_omega") and getattr(agent, "anneal_active", False):
                current_omega = agent.memory.get_current_omega()
                print(f"  ReaPER omega: {current_omega:.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_every == 0:
            base_dir = os.path.join('checkpoints', f'{num_qubits}_qubit')
            os.makedirs(base_dir, exist_ok=True)
            # print(anneal_active)
            # exit()
            agent.save(os.path.join(base_dir, f'dqn_{replay}_hrc_seed_{seed}_anneal_{anneal_active}_ep{episode}.pth'))
            
            save_episode_metrics(episode_data, episode + 1, 'hrc', f'dqn_{replay}', anneal_active, env.num_qubits, seed)
            
            # NEW: Save evaluation results
            if eval_results:
                save_eval_results(eval_results, replay, anneal_active, env.num_qubits, seed)
    
    # Final save
    save_episode_metrics(episode_data, num_episodes, 'hrc', f'dqn_{replay}', anneal_active, env.num_qubits, seed, final=True)
    if eval_results:
        save_eval_results(eval_results, replay, anneal_active, env.num_qubits, seed, final=True)
    
    return episode_rewards, episode_lengths, episode_fidelities, episode_data, eval_results


def save_episode_metrics(episode_data, episode_num, gateset, agent_type, anneal_active, num_qubits, seed, final=False):
    """Save episode-level metrics""" 
    
    base_dir = os.path.join('results', f'{num_qubits}_qubit')
    os.makedirs(base_dir, exist_ok=True)
    
    if final:
        pkl_path = os.path.join(
            base_dir,
            f'episode_metrics_final_gateset_{gateset}_agent_{agent_type}_'
            f'anneal_{anneal_active}_seed_{seed}.pkl'
        )
        print(f"\nSaving final episode metrics to {pkl_path}")
        print(f"Total episodes recorded: {len(episode_data)}")
    else:
        pkl_path = os.path.join(
            base_dir,
            f'episode_metrics_gateset_{gateset}_agent_{agent_type}_'
            f'anneal_{anneal_active}_seed_{seed}.pkl'
        )
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(episode_data, f)


def save_eval_results(eval_results, replay, anneal_active, num_qubits, seed, final=False):
    """Save evaluation results"""
    import pickle
    
    base_dir = os.path.join('results', f'{num_qubits}_qubit')
    os.makedirs(base_dir, exist_ok=True)
    
    if final:
        pkl_name = f'eval_results_final_seed_dqn_{replay}_anneal_{anneal_active}_{seed}.pkl'
        csv_name = f'eval_results_final_seed_dqn_{replay}_anneal_{anneal_active}_{seed}.csv'
    else:
        pkl_name = f'eval_results_seed_dqn_{replay}_anneal_{anneal_active}_{seed}.pkl'
        csv_name = f'eval_results_seed_dqn_{replay}_anneal_{anneal_active}_{seed}.csv'
    
    pkl_path = os.path.join(base_dir, pkl_name)
    csv_path = os.path.join(base_dir, csv_name)

    with open(pkl_path, 'wb') as f:
        pickle.dump(eval_results, f)
    
    # Also save as CSV for easy viewing
    df = pd.DataFrame(eval_results)
    df.to_csv(csv_path, index=False)
    print(f"Evaluation results saved to {csv_path}")


def set_global_seed(seed=42):
    """Set global seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")


# ========== Main ==========

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Train DQN+replay buffer for HRC quantum compilation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--episodes', type=int, default=100000, help='Number of episodes')
    parser.add_argument('--print-every', type=int, default=1000, help='Print frequency')
    parser.add_argument('--save-every', type=int, default=10000, help='Save frequency')
    parser.add_argument('--eval-every', type=int, default=1000, help='Evaluation frequency')
    parser.add_argument('--eval-episodes', type=int, default=100, help='Episodes per evaluation')
    parser.add_argument('--replay', type=str, default='her', help='Exeorience replay type: her or per or reaper or reaper_anneal')
    args = parser.parse_args()
    
    SEED = args.seed
    set_global_seed(SEED)

    if args.replay == 'per':
        buffer = 'PER'
    elif args.replay == 'her':
        buffer = 'HER'
    elif args.replay == 'reaper':
        buffer = 'ReaPER'
    elif args.replay == 'reaper_anneal':
        buffer = 'ReaPER+'

    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Algorithm: DQN with DQN+{buffer} buffer")
    print(f"Gateset: HRC (Harrow-Recht-Chuang)")
    print(f"Reward: Sparse")
    print(f"Seed: {SEED}")
    print(f"{'='*60}\n")
    
    # Environment config (Table 4)
    env_config = {
        'env': {
            'num_qubits': 2,
            'max_episode_length': 350,  # Table 4
            'tolerance': 0.99,
            'gateset': 'hrc',  # HRC gates
            'mode': 'paper',  # Use paper's observation method
            # NO 'reward_type' here - environment will auto-set to 'sparse' for HRC
        }
    }
    
    if args.replay == 'her':
        agent_config = {
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.99995,  # Table 4
            'learning_rate': 3e-4,  # Table 4: 0.0001
            'reaper_omega_anneal': False,
            'batch_size': 200,  # Table 4
            'memory_size': 500000,  # Table 4: 5·10^5 experiences
            'target_update_freq': 100,  # Update every episode
            'her_k': 5,  # Table 4 default
            'her_strategy': 'future',  # Sample from future states
        }
    elif args.replay == 'per':
        agent_config = {
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.99995,  # Table 4
            'reaper_omega_anneal': False,
            'learning_rate': 3e-4,  # Table 4: 0.0001
            'batch_size': 200,  # Table 4
            'memory_size': 500000,  # Table 4: 5·10^5 experiences
            'target_update_freq': 100,  # Update every episode
            'her_k': 5,  # Table 4 default
            'her_strategy': 'future',  # Sample from future states
        }
    elif args.replay == 'reaper':
        agent_config = {
                    'gamma': 0.99,
                    'learning_rate': 3e-4,
                    'batch_size': 350,
                    'memory_size': 500000,
                    
                    
                    # ReaPER with annealing
                    'reaper_alpha': 0.4,
                    'reaper_omega_anneal': False,      # ← ENABLE ANNEALING
                    'reaper_omega_start': 0.6,        # Start high
                    'reaper_omega_end': 0.15,          # End low
                    'reaper_omega_frames': 1000000,     # Anneal over first X frames
                    'reaper_beta_start': 0.4,
                    'reaper_beta_frames': 100000,
                }
    elif args.replay == 'reaper_anneal':
        agent_config = {
                    'gamma': 0.99,
                    'learning_rate': 3e-4,
                    'batch_size': 350,
                    'memory_size': 500000,
                    'epsilon_decay': 0.99997,  # Table 4
                    
                    # ReaPER with annealing
                    'reaper_alpha': 0.4,
                    'reaper_omega_anneal': True,      # ← ENABLE ANNEALING
                    'reaper_omega_start': 0.1,        # Start high
                    'reaper_omega_end': 0.7,          # End low
                    'reaper_omega_frames': 500000,     # Anneal over first X frames
                    'reaper_beta_start': 0.4,
                    'reaper_beta_frames': 100000,
                }
    
    # Training config
    training_config = {
        'num_episodes': 50000,
        'max_steps_per_episode': 350,  # Table 4
        'print_every': 1,
        'save_every': 100,
        'eval_every': 1000,  # NEW
        'eval_episodes': 1000,  # NEW
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create environment
    env = CircuitEnv(env_config, device)
    
    print("=== Environment Info ===")
    print(f"Gateset: {env.gateset}")
    print(f"Reward type: {env.reward_type}")  # Should be 'sparse'
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_size}")
    print(f"Max episode length: {env.max_episode_length}")
    print(f"Tolerance: {env.tolerance}")
    print(f"Gates: {env.gate_names}\n")
    

    if args.replay == 'per':
        print("Creating DQN+PER agent...\n")
        # Create DQN+PER agent
        agent = DQNAgentWithPER(env.state_size, env.action_size, agent_config, device, network_type='paper', seed=SEED)
    
    elif args.replay == 'reaper':
        print("Creating DQN+REAPER agent...\n")
        # Create DQN+REAPER agent
        agent = DQNAgentWithReaPER(env.state_size, env.action_size, agent_config, device, network_type='paper', seed=SEED)
    elif args.replay == 'reaper_anneal':
        print("Creating DQN+REAPER agent...\n")
        # Create DQN+REAPER agent
        agent = DQNAgentWithReaPER(env.state_size, env.action_size, agent_config, device, network_type='paper', seed=SEED)
    
    elif args.replay == 'her':
        print("Creating DQN+HER agent...\n")
        # Create DQN+HER agent
        agent = DQNAgentWithHER(env.state_size, env.action_size, agent_config, device, seed=SEED)
    
    else:
        print(f"ERROR: Unknown replay type: '{args.replay}'")
        print(f"Expected 'per' or 'her' or 'reaper'")
        exit()
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    # Train
    episode_rewards, episode_lengths, episode_fidelities, episode_data, eval_results = train_dqn(
        env, agent, args.replay, agent_config['reaper_omega_anneal'], training_config, env.num_qubits, seed=SEED
    )
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Memory size: {len(agent.memory)}")
    print(f"Total training steps: {agent.steps_done}")
    
    if eval_results:
        print(f"\n{'='*60}")
        print("FINAL EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        final_eval = eval_results[-1]
        print(f"Final Success Rate: {final_eval['success_rate']:.2%}")
        print(f"Final Mean Fidelity: {final_eval['mean_fidelity']:.6f}")
        print(f"Final Mean Length: {final_eval['mean_length']:.1f} ± {final_eval['std_length']:.1f}")
        
        # Show learning curve
        print(f"\nLearning Curve:")
        for i, result in enumerate(eval_results):
            ep = result['episode']
            sr = result['success_rate']
            mf = result['mean_fidelity']
            print(f"  Episode {ep:6d}: Success Rate = {sr:5.1%}, Mean Fidelity = {mf:.6f}")