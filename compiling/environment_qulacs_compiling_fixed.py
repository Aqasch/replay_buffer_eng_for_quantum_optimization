import torch
from sys import stdout
import numpy as np
import copy


from qiskit import QuantumCircuit
from qiskit_aer import Aer





class CircuitEnv():


    def __init__(self, conf, device):
        self.num_qubits = conf['env']['num_qubits']
        self.max_episode_length = conf['env'].get('max_episode_length', 130)
        self.tolerance = conf['env'].get('tolerance', 0.99)
        self.ham_type = 'quantum_compile'
        self.gateset = conf['env']['gateset']
        self.last_gate_label = None


        self.mode = conf['env'].get('mode', 'paper')  # 'paper' or 'matrix'
        self.basis_type = 'clifford+T'
        self.reward_type = conf['env'].get('reward_type', 'dense')
        self.device = device


        # Initialize action space (Clifford + T)
        self._init_action_space()


        # === Fixed Gridsynth target: Rz(pi/128) ===
        theta = np.pi / 128.0
        self.target_unitary = np.array([
            [np.exp(-1j * theta / 2.0), 0.0],
            [0.0, np.exp(1j * theta / 2.0)]
        ], dtype=np.complex128)


        # State size for observation
        dim = 2 ** self.num_qubits
        self.state_size = 2 * dim * dim  # Flattened real/imag parts


        # Tracking
        self.step_counter = -1
        self.prev_fidelity = None
        self.action_list = []
        self.current_unitary = None


        stdout.flush()


    def _init_action_space(self):
        """Initialize action space for single-qubit Clifford+T gates."""
        # Minimal set: H, S, S†, T, T†
        # You can drop S†/T† if you prefer, but this matches a richer word set.
        # self.action_to_gate = {
        #     0: 'H',
        #     1: 'S',
        #     2: 'Sdg',
        #     3: 'T',
        #     4: 'Tdg',
        # }
        self.action_to_gate = {
            0: 'H',
            1: 'S',
            # 2: 'Sdg',
            2: 'T',
            # 4: 'Tdg',
            3: 'X'
        }
        # self.gate_names = ['H', 'S', 'S†', 'T', 'T†']
        self.gate_names = ['H', 'S', 'T', 'X']
        self.action_size = len(self.action_to_gate)


    def _matrix_to_state_vector(self, matrix):
        """Convert complex matrix to real-valued state vector."""
        flat = matrix.flatten()
        state = np.concatenate([flat.real, flat.imag])
        return state.astype(np.float32)


    def _get_observation(self):
        """Get observation based on mode."""
        if self.mode == 'matrix':
            # Mode 1: Direct current unitary
            obs_matrix = self.current_unitary
        else:
            # Mode 2: O_n = U_target^† @ U_current
            obs_matrix = self.target_unitary.conj().T @ self.current_unitary


        return self._matrix_to_state_vector(obs_matrix)

    def select_action(self, state, training=True):
        # epsilon-greedy
        if training and random.random() < self.epsilon:
            # random action, but avoid H after H if possible
            if hasattr(self.env, "last_gate_label") and self.env.last_gate_label == "H":
                # list of action indices whose gate is not 'H'
                valid_actions = [
                    a for a in range(self.action_size)
                    if self.env.action_to_gate[a] != "H"
                ]
                if valid_actions:
                    return random.choice(valid_actions)
            return random.randrange(self.action_size)

        # greedy
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0)).squeeze(0)

        if hasattr(self.env, "last_gate_label") and self.env.last_gate_label == "H":
            # mask all H-actions
            for a in range(self.action_size):
                if self.env.action_to_gate[a] == "H":
                    q_values[a] = -1e9

        action = int(torch.argmax(q_values).item())
        return action



    def step(self, action, train_flag=True):
        self.step_counter += 1
        self.action_list.append(action)


        gate_label = self.action_to_gate[action]

        gate_label = self.action_to_gate[action]
        gate_matrix = self._get_gate_matrix(gate_label)
        self.current_unitary = gate_matrix @ self.current_unitary
        self.last_gate_label = gate_label
        # gate_matrix = self._get_gate_matrix(gate_label)
        # self.current_unitary = gate_matrix @ self.current_unitary


        next_state = self._get_observation()


        # Average gate fidelity (unchanged)
        fidelity = self.average_gate_fidelity()


        # Operator norm error
        op_err = self.operator_norm_error()


        # Termination: use operator norm epsilon instead of AGF tolerance
        # e.g., self.epsilon set from config['env']['epsilon']
        eps = getattr(self, "epsilon", None)
        if eps is not None:
            error_achieved = (op_err <= eps)
        else:
            # fallback to old fidelity tolerance if epsilon not set
            error_achieved = (fidelity >= self.tolerance)


        max_steps_reached = self.step_counter >= (self.max_episode_length - 1)
        done = error_achieved or max_steps_reached
        if done:
            print(f"Episode {episode+1} done: steps={step_in_episode}, solved={info['solved']}, op_err={info['operator_norm_error']:.6e}, fidelity={info['fidelity']:.6f}")


        # Define "error" attribute for reward: choose one metric
        # For training, you probably want the more sensitive operator norm.
        if eps is not None:
            self.error = op_err
        else:
            self.error = 1.0 - fidelity


        reward = self.compute_reward(fidelity, error_achieved)


        self.prev_fidelity = fidelity


        return (torch.tensor(next_state, dtype=torch.float32, device=self.device),
                torch.tensor(reward, dtype=torch.float32, device=self.device),
                done)



    def reset(self):
        """Reset environment for new episode."""
        # Reset tracking
        # self.step_counter = -1
        # self.action_list = []

        self.step_counter = -1
        self.action_list = []
        self.last_gate_label = None


        # Start from identity
        dim = 2 ** self.num_qubits
        self.current_unitary = np.eye(dim, dtype=np.complex128)


        # Initial fidelity
        self.prev_fidelity = self.average_gate_fidelity()


        # Get initial observation
        state = self._get_observation()


        return torch.tensor(state, dtype=torch.float32, device=self.device)


    def _get_gate_matrix(self, gate_label):
        """Get 2x2 matrix for single-qubit Clifford+T gate."""
        if gate_label == 'H':
            return (1.0 / np.sqrt(2.0)) * np.array(
                [[1.0,  1.0],
                 [1.0, -1.0]],
                dtype=np.complex128
            )
        elif gate_label == 'S':
            return np.array(
                [[1.0, 0.0],
                 [0.0, 1.0j]],
                dtype=np.complex128
            )
        elif gate_label == 'Sdg':
            return np.array(
                [[1.0, 0.0],
                 [0.0, -1.0j]],
                dtype=np.complex128
            )
        elif gate_label == 'T':
            return np.array(
                [[1.0, 0.0],
                 [0.0, np.exp(1j * np.pi / 4.0)]],
                dtype=np.complex128
            )
        elif gate_label == 'Tdg':
            return np.array(
                [[1.0, 0.0],
                 [0.0, np.exp(-1j * np.pi / 4.0)]],
                dtype=np.complex128
            )
        elif gate_label == 'X':
            return np.array(
                [[0.0, 1.0],
                [1.0, 0.0]],
                dtype=np.complex128
            )
        else:
            raise ValueError(f"Unknown gate label: {gate_label}")


    def make_circuit(self):
        """Construct Qiskit circuit from action list (for visualization)."""
        circuit = QuantumCircuit(self.num_qubits)


        for action in self.action_list:
            gate_label = self.action_to_gate[action]
            if gate_label == 'H':
                circuit.h(0)
            elif gate_label == 'S':
                circuit.s(0)
            elif gate_label == 'Sdg':
                circuit.sdg(0)
            elif gate_label == 'T':
                circuit.t(0)
            elif gate_label == 'Tdg':
                circuit.tdg(0)


        return circuit



    def operator_norm_error(self):
        """
        Operator norm distance ||U_target - U_current||_2.
        """
        diff = self.target_unitary - self.current_unitary
        # For 2x2, full SVD is cheap
        svals = np.linalg.svd(diff, compute_uv=False)
        return float(svals[0])



    def average_gate_fidelity(self):
        """
        Compute average gate fidelity between target and current unitaries
        AGF = (|Tr(U_target^† @ U_current)|^2 + d) / (d(d+1))
        """
        d = self.current_unitary.shape[0]
        trace = np.trace(self.target_unitary.conj().T @ self.current_unitary)
        fidelity = (np.abs(trace) ** 2 + d) / (d * (d + 1))
        return float(fidelity)


    def compute_reward(self, fidelity, error_achieved):
        """Compute reward based on reward type."""
        if self.reward_type == 'dense':
            L = self.max_episode_length
            n = self.step_counter


            if error_achieved:
                reward = (L - n + 1)
            else:
                # self.error is either (1 - fidelity) or operator-norm distance
                reward = - self.error / L


        elif self.reward_type == 'sparse':
            L = self.max_episode_length
            if error_achieved:
                reward = 0.0
            else:
                reward = -1.0 / L
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")


        return reward


    def get_episode_info(self):
        """Get information about current episode."""
        fidelity = self.average_gate_fidelity()
        circuit = self.make_circuit()


        # T-count: number of T/T† gates
        t_count = sum(self.gate_names[a].startswith('T') for a in self.action_list)


        return {
            'fidelity': fidelity,
            'steps': self.step_counter + 1,
            'sequence_length': len(self.action_list),
            't_count': t_count,
            'action_sequence': self.action_list.copy(),
            'gate_names': [self.gate_names[a] for a in self.action_list],
            'solved': fidelity >= self.tolerance,
            'circuit': circuit,
            'error': 1.0 - fidelity
        }



if __name__ == "__main__":
    # Example usage
    config = {
        'env': {
            'num_qubits': 1,
            'max_episode_length': 130,
            # For eps = 1e-10:
            # 'tolerance': 1.0 - 1e-10,
            'tolerance': 0.9999,
            'reward_type': 'dense',
            'mode': 'paper',  # Use paper's observation method
            'curriculum_type': 'VanillaCurriculum',
            'thresholds': [0.99],
            'switch_episodes': [10000],
            'accept_err': 0.01
        }
    }


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = CircuitEnv(config, device)


    print(f"Environment initialized:")
    print(f"  State size: {env.state_size}")
    print(f"  Action size: {env.action_size}")
    print(f"  Gates: {env.gate_names}")
    print(f"  Target unitary (Rz(pi/128)):\n{env.target_unitary}")


    # Test episode
    state = env.reset()
    print(f"\nState shape: {state.shape}")


    done = False
    step_count = 0
    while not done and step_count < 100:
        action = np.random.randint(0, env.action_size)
        next_state, reward, done = env.step(action)
        step_count += 1


    info = env.get_episode_info()
    print(f"\n=== Episode Summary ===")
    print(f"Steps: {info['steps']}")
    print(f"Final fidelity: {info['fidelity']:.12f}")
    print(f"Solved: {info['solved']}")
    print(f"T-count: {info['t_count']}")
    print(f"Gate sequence: {' -> '.join(info['gate_names'])}")