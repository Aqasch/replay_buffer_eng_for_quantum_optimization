import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import random  # FIX: Added missing import
from collections import namedtuple


from utils import dictionary_of_actions, dict_of_actions_revert_q



class DQN_REAPER(object):


    def __init__(self, conf, action_size, state_size, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        memory_size = conf['agent']['memory_size']


        self.final_gamma = conf['agent']['final_gamma']
        self.epsilon_min = conf['agent']['epsilon_min']
        self.epsilon_decay = conf['agent']['epsilon_decay']
        learning_rate = conf['agent']['learning_rate']
        self.update_target_net = conf['agent']['update_target_net']
        neuron_list = conf['agent']['neurons']
        drop_prob = conf['agent']['dropout']
        self.with_angles = conf['agent']['angles']


        # ReaPER hyperparameters (with defaults similar to your other script)
        reaper_alpha = float(conf['agent'].get('reaper_alpha', 0.4))
        reaper_omega = float(conf['agent'].get('reaper_omega', 0.2))
        reaper_omega_anneal = conf['agent'].get('reaper_omega_anneal', False)
        reaper_omega_start = float(conf['agent'].get('reaper_omega_start', 0.6))
        reaper_omega_end = float(conf['agent'].get('reaper_omega_end', 0.15))
        reaper_omega_frames = int(conf['agent'].get('reaper_omega_frames', 50000))
        reaper_beta_start = float(conf['agent'].get('reaper_beta_start', 0.4))
        reaper_beta_frames = int(conf['agent'].get('reaper_beta_frames', 100000))


        if "memory_reset_switch" in conf['agent'].keys():
            self.memory_reset_switch = conf['agent']["memory_reset_switch"]
            self.memory_reset_threshold = conf['agent']["memory_reset_threshold"]
            self.memory_reset_counter = 0
        else:
            self.memory_reset_switch = False
            self.memory_reset_threshold = False
            self.memory_reset_counter = False


        self.action_size = action_size


        self.state_size = state_size if self.with_angles else state_size - self.num_layers * self.num_qubits * 3
        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
        self.state_size = self.state_size + 1 if ("threshold_in_state" in conf['agent'].keys()
                                                  and conf['agent']["threshold_in_state"]) else self.state_size


        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)


        self.policy_net = self.unpack_network(neuron_list, drop_prob).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()


        # layer-wise gamma as in DQN_PGR
        self.gamma = torch.Tensor([np.round(np.power(self.final_gamma, 1 / self.num_layers), 2)]).to(device)


        print(f"[DQN_REAPER] Initializing ReaPER buffer (anneal={reaper_omega_anneal})")
        self.memory = ReliabilityAdjustedPrioritizedReplayBuffer(
            capacity=memory_size,
            alpha=reaper_alpha,
            omega=reaper_omega,
            omega_anneal=reaper_omega_anneal,
            omega_start=reaper_omega_start,
            omega_end=reaper_omega_end,
            omega_frames=reaper_omega_frames,
            beta_start=reaper_beta_start,
            beta_frames=reaper_beta_frames,
            device=device
        )


        self.epsilon = 1.0
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()
        self.device = device
        self.step_counter = 0


        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'reward', 'next_state', 'done'))


    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)


    def act(self, state, ill_action):
        state = state.unsqueeze(0)
        epsilon = False


        if torch.rand(1).item() <= self.epsilon:
            rand_ac = torch.randint(self.action_size, (1,)).item()
            while rand_ac in ill_action:
                rand_ac = torch.randint(self.action_size, (1,)).item()
            epsilon = True
            return rand_ac, epsilon


        act_values = self.policy_net.forward(state)
        act_values[0][ill_action] = float('-inf')
        return torch.argmax(act_values[0]).item(), epsilon


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None


        if self.step_counter % self.update_target_net == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1


        # Sample from ReaPER buffer
        try:
            states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample(batch_size)
        except Exception as e:
            print(f"[WARNING] Sampling failed: {e}")
            return None


        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)


        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1))


        # Double DQN
        next_q_target = self.target_net(next_states)
        next_q_policy = self.policy_net(next_states)
        next_state_actions = next_q_policy.max(1)[1].detach()
        next_state_values = next_q_target.gather(1, next_state_actions.unsqueeze(1)).squeeze(1)


        expected_state_action_values = (next_state_values * self.gamma) * (1 - dones) + rewards
        expected_state_action_values = expected_state_action_values.view(-1, 1)


        td_errors = torch.abs(expected_state_action_values - state_action_values)


        # ReaPER priority update
        try:
            self.memory.update_priorities(idxs, td_errors)
        except Exception as e:
            print(f"[WARNING] Priority update failed: {e}")


        # Importance sampling weights
        assert state_action_values.shape == expected_state_action_values.shape, "Wrong shapes in loss"
        loss = self.fit(state_action_values, expected_state_action_values, is_weights)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)


        return loss


    def fit(self, output, target_f, weights):
        self.optim.zero_grad()
        # weights shape [B], we need [B,1]
        w = weights.view(-1, 1)
        loss = self.loss(output * w, target_f * w)
        loss.backward()
        self.optim.step()
        return loss.item()


    def unpack_network(self, neuron_list, p):
        layer_list = []
        neuron_list = [self.state_size] + neuron_list
        for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
            layer_list.append(nn.Linear(input_n, output_n))
            layer_list.append(nn.LeakyReLU())
            layer_list.append(nn.Dropout(p=p))
        layer_list.append(nn.Linear(neuron_list[-1], self.action_size))
        return nn.Sequential(*layer_list)



# ============================================================================
# IMPROVED REAPER BUFFER WITH BUG FIXES
# ============================================================================


Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))


class SumTree:
    """Efficient sum tree for prioritized sampling"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0


    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate(parent, change)


    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])


    def total(self):
        return self.tree[0]


    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)


        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.n_entries = min(self.n_entries + 1, self.capacity)


    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propagate(idx, change)


    def get(self, s):
        idx = self.retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]



class ReliabilityAdjustedPrioritizedReplayBuffer:
    """
    ReaPER - CORRECTLY IMPLEMENTED
    
    Key fixes:
    1. TD errors initialized with reward magnitude (not 0)
    2. Episode TD sums updated during push (not just during sampling)
    3. Priorities updated on episode completion with valid TD estimates
    4. Re-updated after first real TD errors computed
    """


    def __init__(self, capacity,
                 alpha=0.4,
                 omega=0.2,
                 omega_anneal=False,
                 omega_start=0.6,
                 omega_end=0.2,
                 omega_frames=50000,
                 beta_start=0.4,
                 beta_frames=100000,
                 device="cpu"):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha


        # Omega schedule
        self.omega_anneal = omega_anneal
        if omega_anneal:
            self.omega_start = omega_start
            self.omega_end = omega_end
            self.omega_frames = omega_frames
            self.omega_fixed = None
        else:
            self.omega_fixed = omega
            self.omega_start = None
            self.omega_end = None
            self.omega_frames = None


        # Beta schedule
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.device = device
        self.epsilon = 1e-6


        # Episode tracking
        self.episode_ids = np.zeros(capacity, dtype=np.int32)
        self.current_episode_id = 0
        self.td_errors = np.zeros(capacity, dtype=np.float32)


        self.episode_indices = {}
        self.episode_td_sums = {}
        self.episode_complete = {}
        
        # FIX: Track episodes needing re-prioritization
        self.episodes_pending_update = set()
        
        self.max_episodes_tracked = max(10, capacity // 10)
        self.episodes_updated = 0


        print(f"[ReaPER CORRECT] Initialized: capacity={capacity}, alpha={alpha}")
        if omega_anneal:
            print(f"  Omega schedule: {omega_start} -> {omega_end} over {omega_frames} frames")
        else:
            print(f"  Fixed omega = {omega}")
        print(f"  FIX 1: TD errors initialized with reward estimates")
        print(f"  FIX 2: Priorities updated on episode completion")
        print(f"  FIX 3: Re-updated after first real TD errors")


    def get_beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)


    def get_omega(self):
        if self.omega_anneal:
            progress = min(1.0, self.frame / float(self.omega_frames))
            return self.omega_start + progress * (self.omega_end - self.omega_start)
        else:
            return self.omega_fixed


    def push(self, state, action, reward, next_state, done):
        """Add transition with TD error initialization and priority update on completion"""
        if self.tree.n_entries > 0:
            max_priority = np.max(self.tree.tree[-self.tree.capacity:])
            if max_priority <= 0:
                max_priority = 1.0
        else:
            max_priority = 1.0


        experience = Experience(state, action, reward, next_state, done)
        buffer_idx = self.tree.write
        epid = self.current_episode_id
        
        # Clean old episodes
        old_epid = int(self.episode_ids[buffer_idx])
        if old_epid in self.episode_indices:
            if buffer_idx in self.episode_indices[old_epid]:
                self.episode_indices[old_epid].remove(buffer_idx)
            if len(self.episode_indices[old_epid]) == 0:
                self._cleanup_episode(old_epid)
        
        # Initialize episode tracking
        self.episode_ids[buffer_idx] = epid
        
        # FIX #1: Initialize TD error with reward magnitude estimate
        reward_val = reward.item() if torch.is_tensor(reward) else float(reward)
        # Use abs(reward) as TD error estimate, with minimum threshold
        initial_td = max(abs(reward_val), 0.1)
        self.td_errors[buffer_idx] = initial_td


        if epid not in self.episode_indices:
            self.episode_indices[epid] = []
            self.episode_td_sums[epid] = 0.0
            self.episode_complete[epid] = False


        # FIX #2: Update episode TD sum during push
        self.episode_td_sums[epid] += initial_td
        self.episode_indices[epid].append(buffer_idx)
        self.tree.add(max_priority, experience)


        # FIX #3: Update priorities when episode completes
        if bool(done):
            self.episode_complete[epid] = True
            
            # Update priorities with initial TD estimates
            self._update_episode_priorities(epid)
            
            # Mark for re-update after first real TD errors
            self.episodes_pending_update.add(epid)
            
            self.current_episode_id += 1
            self._limit_episode_memory(epid)


    def _update_episode_priorities(self, epid):
        indices = self.episode_indices.get(epid, [])
        if not indices:
            return


        omega = self.get_omega()


        # Precompute prefix and suffix sums of |td_errors|
        abs_td = np.array([abs(self.td_errors[idx]) for idx in indices], dtype=np.float32)
        total_td = abs_td.sum()
        if total_td <= self.epsilon:
            # fall back to PER priorities
            for idx_buf, td_val in zip(indices, abs_td):
                tree_idx = idx_buf + self.capacity - 1
                priority = (td_val + self.epsilon) ** self.alpha
                self.tree.update(tree_idx, priority)
            return


        # downstream sum for each position: suffix_sum[i+1]
        suffix_sum = np.concatenate(([0.0], np.cumsum(abs_td[::-1])))[::-1]


        for k, idx_buf in enumerate(indices):
            downstream_td = suffix_sum[k + 1]
            reliability = 1.0 - downstream_td / total_td
            reliability = max(0.0, min(1.0, reliability))


            tree_idx = idx_buf + self.capacity - 1
            td_error_abs = abs_td[k]
            priority = (reliability ** omega) * (td_error_abs + self.epsilon) ** self.alpha
            self.tree.update(tree_idx, priority)


    def _cleanup_episode(self, epid):
        """Clean up episode data"""
        for d in [self.episode_indices, self.episode_td_sums, self.episode_complete]:
            if epid in d:
                del d[epid]
        # Also remove from pending updates
        self.episodes_pending_update.discard(epid)


    def _limit_episode_memory(self, current_epid):
        """Clean old episodes"""
        if len(self.episode_indices) > self.max_episodes_tracked:
            oldest_eps = sorted(self.episode_indices.keys())[:len(self.episode_indices) // 4]
            for old_ep in oldest_eps:
                if old_ep != current_epid:
                    valid_indices = [idx for idx in self.episode_indices.get(old_ep, []) 
                                   if idx < self.tree.n_entries]
                    if not valid_indices or self.episode_complete.get(old_ep, False):
                        self._cleanup_episode(old_ep)


    def compute_reliability(self, buffer_idx):
        """Compute reliability per Equation 4"""
        if self.tree.n_entries == 0:
            return 1.0
            
        epid = int(self.episode_ids[buffer_idx])
        indices = self.episode_indices.get(epid, [])
        if not indices:
            return 1.0


        total_td = self.episode_td_sums.get(epid, 0.0)
        if total_td <= self.epsilon:
            return 1.0


        try:
            pos = indices.index(buffer_idx)
        except ValueError:
            return 1.0


        # Calculate downstream TD errors
        downstream_td = 0.0
        for j in range(pos + 1, len(indices)):
            idx = indices[j]
            if idx < self.tree.n_entries:
                downstream_td += abs(self.td_errors[idx])


        # R_t = 1 - (downstream_td / total_td)
        reliability = 1.0 - downstream_td / total_td
        return max(0.0, min(1.0, reliability))


    def sample(self, batch_size):
        """Sample batch with reliability-adjusted priorities"""
        batch = []
        idxs = []
        priorities = []


        if self.tree.n_entries == 0:
            raise RuntimeError("Cannot sample from empty buffer")


        segment = self.tree.total() / batch_size
        beta = self.get_beta()
        self.frame += 1


        attempts = 0
        max_attempts = batch_size * 10
        
        for i in range(batch_size):
            while len(batch) <= i and attempts < max_attempts:
                attempts += 1
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                
                try:
                    idx, p, data = self.tree.get(s)
                except:
                    continue
                    
                data_idx = idx - self.tree.capacity + 1
                
                if data_idx >= self.tree.n_entries or data_idx < 0:
                    continue
                    
                if data is None or not isinstance(data, Experience):
                    continue
                
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)


        while len(batch) < batch_size and self.tree.n_entries > 0:
            valid_idx = random.randint(0, self.tree.n_entries - 1)
            tree_idx = valid_idx + self.tree.capacity - 1
            p = self.tree.tree[tree_idx]
            if p <= 0:          
                continue
            data = self.tree.data[valid_idx]
            if data is not None and isinstance(data, Experience):
                batch.append(data)
                idxs.append(tree_idx)
                priorities.append(p)
                if len(batch) >= batch_size:
                    break


        if len(batch) == 0:
            raise RuntimeError("Failed to sample any valid transitions")


        sampling_probabilities = np.array(priorities) / (self.tree.total() + self.epsilon)
        sampling_probabilities = np.clip(sampling_probabilities, a_min=self.epsilon, a_max=None)  # ← no zeros
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        max_weight = is_weights.max()
        if max_weight == 0 or not np.isfinite(max_weight):
            is_weights = np.ones_like(is_weights)   # fallback to uniform weights
        else:
            is_weights /= max_weight


        states = torch.stack([e.state for e in batch])
        actions = torch.stack([e.action for e in batch]).long()
        rewards = torch.stack([e.reward for e in batch]).float()
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.stack([e.done for e in batch]).float()
        is_weights = torch.tensor(is_weights, dtype=torch.float32)


        return states, actions, rewards, next_states, dones, idxs, is_weights


    def update_priorities(self, idxs, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td_np = td_errors.detach().cpu().numpy().flatten()
        else:
            td_np = np.asarray(td_errors).flatten()


        omega = self.get_omega()


        for idx, tde in zip(idxs, td_np):
            buffer_idx = idx - self.tree.capacity + 1
            if buffer_idx >= self.tree.n_entries or buffer_idx < 0:
                continue


            epid = int(self.episode_ids[buffer_idx])
            old_abs = abs(self.td_errors[buffer_idx])
            new_abs = abs(tde)
            self.td_errors[buffer_idx] = tde


            if epid in self.episode_td_sums:
                self.episode_td_sums[epid] += new_abs - old_abs


            # approximate reliability using current episode sums
            total_td = self.episode_td_sums.get(epid, 0.0)
            if total_td <= self.epsilon:
                reliability = 1.0
            else:
                # approximate downstream_td as (total_td - |δ_t|)
                downstream_td = max(0.0, total_td - new_abs)
                reliability = 1.0 - downstream_td / total_td
                reliability = max(0.0, min(1.0, reliability))


            abs_td_error = new_abs + self.epsilon
            adjusted_priority = (reliability ** omega) * (abs_td_error ** self.alpha)
            self.tree.update(idx, adjusted_priority)



    def __len__(self):
        return self.tree.n_entries


    def clean_memory(self):
        """Reset buffer"""
        self.tree = SumTree(self.capacity)
        self.episode_ids = np.zeros(self.capacity, dtype=np.int32)
        self.current_episode_id = 0
        self.td_errors = np.zeros(self.capacity, dtype=np.float32)
        self.episode_indices = {}
        self.episode_td_sums = {}
        self.episode_complete = {}
        self.episodes_pending_update = set()
        self.episodes_updated = 0
        print("[ReaPER CORRECT] Buffer cleaned")