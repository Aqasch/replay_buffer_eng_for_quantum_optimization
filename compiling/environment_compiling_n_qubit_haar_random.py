import torch
import numpy as np
from qiskit import QuantumCircuit
from sys import stdout


class CircuitEnv:

    def __init__(self, conf, device):
        self.num_qubits = conf['env']['num_qubits']
        self.max_episode_length = conf['env'].get('max_episode_length', 130)
        self.tolerance = conf['env'].get('tolerance', 0.99)
        self.ham_type = 'quantum_compile'

        self.mode = conf['env'].get('mode', 'paper')
        self.reward_type = conf['env'].get('reward_type', 'dense')
        self.device = device
        self.gateset = 'Paper'

        # Target generation: exactly as in paper
        self.target_mode = conf['env'].get('target_mode', 'random_composed')
        self.fixed_target_gate = conf['env'].get('fixed_target_gate', None)
        # Paper: 5 < N < 10^4
        self.min_target_gates = conf['env'].get('min_target_gates', 6)  # > 5
        self.max_target_gates = conf['env'].get('max_target_gates', 10000)  # < 10^4

        if self.num_qubits == 1:
            self._init_action_space_1q()
        else:
            self._init_action_space_2q()

        dim = 2 ** self.num_qubits
        self.state_size = 2 * dim * dim

        self.step_counter = -1
        self.prev_fidelity = None
        self.action_list = []
        self.current_unitary = None
        self.target_unitary = None
        self.target_num_gates = None

        stdout.flush()

    def _init_action_space_1q(self):
        self.angles = [np.pi / 128, -np.pi / 128]
        self.action_to_gate = {}
        self.gate_names = []
        self.gate_matrices = {}
        action_id = 0

        for angle in self.angles:
            self.action_to_gate[action_id] = ('1q', 'RZ', 0, angle)
            angle_deg = int(np.degrees(angle))
            self.gate_names.append(f"RZ({angle_deg:+d}°)")
            self.gate_matrices[action_id] = self._rz_matrix(angle)
            action_id += 1

        self.action_size = action_id

    def _init_action_space_2q(self):
        """Gate set B: {XX(±π/128), YY(±π/128), Rz(±π/128)⊗I, I⊗Rz(±π/128)}"""
        self.angles = [np.pi / 128, -np.pi / 128]
        self.action_to_gate = {}
        self.gate_names = []
        self.gate_matrices = {}
        action_id = 0

        I = np.eye(2, dtype=np.complex128)

        # RZ on each qubit
        for q in range(self.num_qubits):
            for angle in self.angles:
                self.action_to_gate[action_id] = ('1q', 'RZ', q, angle)
                angle_deg = int(np.degrees(angle))
                self.gate_names.append(f"RZ({angle_deg:+d}°)_q{q}")
                
                rz = self._rz_matrix_2x2(angle)
                if q == 0:
                    full_gate = np.kron(I, rz)  # I ⊗ RZ
                else:
                    full_gate = np.kron(rz, I)  # RZ ⊗ I
                
                self.gate_matrices[action_id] = full_gate
                action_id += 1

        # XX and YY
        for gate_type in ['XX', 'YY']:
            for angle in self.angles:
                self.action_to_gate[action_id] = ('2q', gate_type, 0, 1, angle)
                angle_deg = int(np.degrees(angle))
                self.gate_names.append(f"{gate_type}({angle_deg:+d}°)_q01")
                
                if gate_type == 'XX':
                    self.gate_matrices[action_id] = self._xx_matrix(angle)
                else:
                    self.gate_matrices[action_id] = self._yy_matrix(angle)
                
                action_id += 1

        self.action_size = action_id
        assert self.action_size == 8, f"Expected 8 actions, got {self.action_size}"
        print(f"\nGate set B (8 unitaries):")
        for i, name in enumerate(self.gate_names):
            print(f"  {i}: {name}")

    def _rz_matrix_2x2(self, theta):
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)

    def _rz_matrix(self, theta):
        return self._rz_matrix_2x2(theta)

    def _xx_matrix(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c,        0,       0,      -1j*s],
            [0,        c,      -1j*s,   0    ],
            [0,       -1j*s,    c,      0    ],
            [-1j*s,    0,       0,       c   ]
        ], dtype=np.complex128)

    def _yy_matrix(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c,        0,       0,       1j*s],
            [0,        c,      -1j*s,    0   ],
            [0,       -1j*s,    c,       0   ],
            [1j*s,     0,       0,       c   ]
        ], dtype=np.complex128)

    def _zz_matrix(self, theta):
        return np.array([
            [np.exp(-1j*theta/2),  0,   0,   0  ],
            [0,   np.exp(1j*theta/2),   0,   0  ],
            [0,   0,   np.exp(1j*theta/2),   0  ],
            [0,   0,   0,   np.exp(-1j*theta/2)]
        ], dtype=np.complex128)

    def _apply_action_to_unitary(self, unitary, action):
        gate = self.gate_matrices[action]
        return unitary @ gate

    def _generate_target_from_gates(self):
        """
        From paper: "targets randomly built by composing N gates extracted 
        by a set of eight unitaries B... N is randomly set at the beginning 
        of each episode so that 5 < N < 10^4"
        """
        # Randomly choose N in range (5, 10000)
        N = np.random.randint(self.min_target_gates, self.max_target_gates + 1)
        
        dim = 2 ** self.num_qubits
        target = np.eye(dim, dtype=np.complex128)
        
        # Compose N random gates from set B
        for _ in range(N):
            random_action = np.random.randint(0, self.action_size)
            target = self._apply_action_to_unitary(target, random_action)
        
        return target, N

    def _generate_fixed_target(self):
        if self.fixed_target_gate == 'ZZ_pi':
            target = self._zz_matrix(np.pi)
            return target, 0
        else:
            raise ValueError(f"Unknown fixed target: {self.fixed_target_gate}")

    def step(self, action, train_flag=True):
        self.step_counter += 1
        self.action_list.append(action)

        self.current_unitary = self._apply_action_to_unitary(self.current_unitary, action)

        next_state = self._get_observation()
        fidelity = self.average_gate_fidelity()

        fidelity_achieved = fidelity >= self.tolerance
        max_steps_reached = self.step_counter >= (self.max_episode_length - 1)
        done = fidelity_achieved or max_steps_reached

        self.error = 1.0 - fidelity
        reward = self.compute_reward(fidelity, fidelity_achieved)
        self.prev_fidelity = fidelity

        return (torch.tensor(next_state, dtype=torch.float32, device=self.device),
                torch.tensor(reward, dtype=torch.float32, device=self.device),
                done)

    def reset(self):
        self.step_counter = -1
        self.action_list = []

        dim = 2 ** self.num_qubits
        self.current_unitary = np.eye(dim, dtype=np.complex128)
        
        if self.target_mode == 'fixed_gate':
            self.target_unitary, self.target_num_gates = self._generate_fixed_target()
        elif self.target_mode == 'random_composed':
            self.target_unitary, self.target_num_gates = self._generate_target_from_gates()
        elif self.target_mode == 'haar':
            self.target_unitary = self._sample_haar_random_unitary()
            self.target_num_gates = None
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        self.prev_fidelity = self.average_gate_fidelity()
        state = self._get_observation()
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def _matrix_to_state_vector(self, matrix):
        flat = matrix.flatten()
        state = np.concatenate([flat.real, flat.imag])
        return state.astype(np.float32)

    def _get_observation(self):
        if self.mode == 'matrix':
            obs_matrix = self.current_unitary
        else:
            obs_matrix = self.target_unitary.conj().T @ self.current_unitary
        return self._matrix_to_state_vector(obs_matrix)

    def _sample_haar_random_unitary(self):
        d = 2 ** self.num_qubits
        Z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        Q, R = np.linalg.qr(Z)
        diag = np.diag(R)
        Lambda = np.diag(diag / np.abs(diag))
        return Q @ Lambda

    def make_circuit(self):
        circuit = QuantumCircuit(self.num_qubits)
        for action in self.action_list:
            desc = self.action_to_gate[action]
            if desc[0] == '1q':
                _, gate_type, q, angle = desc
                circuit.rz(angle, q)
            elif desc[0] == '2q':
                _, gate_type, i, j, angle = desc
                gate_mat = self._xx_matrix(angle) if gate_type == 'XX' else self._yy_matrix(angle)
                circuit.unitary(gate_mat, [i, j], label=gate_type)
        return circuit

    def average_gate_fidelity(self):
        d = self.current_unitary.shape[0]
        trace = np.trace(self.target_unitary.conj().T @ self.current_unitary)
        fidelity = (np.abs(trace) ** 2 + d) / (d * (d + 1))
        return float(fidelity)

    def compute_reward(self, fidelity, fidelity_achieved):
        if self.reward_type == 'dense':
            L = self.max_episode_length
            n = self.step_counter
            if fidelity_achieved:
                reward = (L - n + 1)
            else:
                distance = 1.0 - fidelity
                reward = -distance / L
        elif self.reward_type == 'sparse':
            L = self.max_episode_length
            reward = 0.0 if fidelity_achieved else -1.0 / L
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        return reward

    def get_episode_info(self):
        fidelity = self.average_gate_fidelity()
        circuit = self.make_circuit()
        return {
            'fidelity': fidelity,
            'steps': self.step_counter + 1,
            'sequence_length': len(self.action_list),
            'action_sequence': self.action_list.copy(),
            'gate_names': [self.gate_names[a] for a in self.action_list],
            'solved': fidelity >= self.tolerance,
            'circuit': circuit,
            'error': 1.0 - fidelity,
            'target_num_gates': self.target_num_gates
        }


if __name__ == "__main__":
    # Test target generation with correct range
    config = {
        'env': {
            'num_qubits': 2,
            'max_episode_length': 130,
            'tolerance': 0.99,
            'reward_type': 'dense',
            'mode': 'paper',
            'target_mode': 'random_composed',
            'min_target_gates': 6,      # > 5
            'max_target_gates': 10000,  # < 10^4
        }
    }
    device = torch.device('cpu')
    env = CircuitEnv(config, device)
    
    print(f"\n{'='*60}")
    print("Testing target generation: 5 < N < 10^4")
    print(f"{'='*60}")
    
    # Generate 100 targets and check initial fidelities
    initial_fidelities = []
    target_sizes = []
    
    for i in range(100):
        state = env.reset()
        initial_fid = env.average_gate_fidelity()
        initial_fidelities.append(initial_fid)
        target_sizes.append(env.target_num_gates)
        
        if i < 5:
            print(f"\nEpisode {i}:")
            print(f"  Target composed of N={env.target_num_gates} gates")
            print(f"  Initial fidelity: {initial_fid:.6f}")
    
    print(f"\n{'='*60}")
    print(f"Statistics over 100 episodes:")
    print(f"{'='*60}")
    print(f"Target size (N):")
    print(f"  Min: {np.min(target_sizes)}")
    print(f"  Max: {np.max(target_sizes)}")
    print(f"  Mean: {np.mean(target_sizes):.1f}")
    print(f"  Median: {np.median(target_sizes):.1f}")
    
    print(f"\nInitial fidelity:")
    print(f"  Min: {np.min(initial_fidelities):.6f}")
    print(f"  Max: {np.max(initial_fidelities):.6f}")
    print(f"  Mean: {np.mean(initial_fidelities):.6f}")
    print(f"  Median: {np.median(initial_fidelities):.6f}")
    
    # Count how many are "easy" (>0.9 initial fidelity)
    easy_count = sum(1 for f in initial_fidelities if f > 0.9)
    print(f"\n  Targets with initial fidelity > 0.9: {easy_count}/100 ({easy_count}%)")
    
    if easy_count > 20:
        print("\n⚠ WARNING: Many targets are too easy!")
        print("This happens when N is small. The range 5 < N < 10^4 includes")
        print("many small values. With small angles, small N ≈ identity.")
    else:
        print("\n✓ Target difficulty looks reasonable!")
