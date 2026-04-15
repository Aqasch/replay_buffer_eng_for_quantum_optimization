import numpy as np
import torch
from qiskit import QuantumCircuit
import copy


class CircuitEnv:
    """
    Single-qubit DRL compiling environment for the 'rotation operators' setting.
    Gate set: six small rotations around X, Y, Z axes (±).
    Observation: flattened real/imag parts of O_n = U_target^† @ U_current (mode='paper').
    Reward: dense, as in Eq. (2) of the paper.
    """

    def __init__(self, conf, device):
        self.num_qubits = conf['env'].get('num_qubits', 1)
        self.max_episode_length = conf['env'].get('max_episode_length', 300)
        self.tolerance = conf['env'].get('tolerance', 0.9999)
        self.reward_type = conf['env'].get('reward_type', 'dense')
        self.mode = conf['env'].get('mode', 'paper')  # 'paper' or 'matrix'
        self.device = device

        # Optional operator-norm epsilon
        self.epsilon = conf['env'].get('epsilon', None)

        # Initialize gate set: labels only, actual angle can vary
        self._init_action_space()

        # Target unitary: will be set via generate_target()
        self.target_unitary = np.eye(2, dtype=np.complex128)

        # Observation/state size
        dim = 2 ** self.num_qubits
        self.state_size = 2 * dim * dim  # real+imag of 2x2 matrix

        # Tracking
        self.step_counter = -1
        self.prev_fidelity = None
        self.action_list = []
        self.current_unitary = None
        self.last_gate_label = None

    # ------------------------------------------------------------------
    # Gate set: labels
    # ------------------------------------------------------------------
    def _init_action_space(self):
        """
        Initialize action space: labels for {Rx(+), Rx(-), Ry(+), Ry(-), Rz(+), Rz(-)}.
        The actual rotation angle can be chosen elsewhere.
        """
        self.action_to_gate = {
            0: 'Rx_pos',  # + around x
            1: 'Rx_neg',  # - around x
            2: 'Ry_pos',  # + around y
            3: 'Ry_neg',  # - around y
            4: 'Rz_pos',  # + around z
            5: 'Rz_neg',  # - around z
        }
        self.gate_names = ['Rx(+)', 'Rx(-)', 'Ry(+)', 'Ry(-)', 'Rz(+)', 'Rz(-)']
        self.action_size = len(self.action_to_gate)

    # ------------------------------------------------------------------
    # Generic single-qubit small rotations
    # ------------------------------------------------------------------
    @staticmethod
    def _rx(alpha):
        c = np.cos(alpha / 2.0)
        s = np.sin(alpha / 2.0)
        return np.array([[c, -1j * s],
                         [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def _ry(alpha):
        c = np.cos(alpha / 2.0)
        s = np.sin(alpha / 2.0)
        return np.array([[c, -s],
                         [s, c]], dtype=np.complex128)

    @staticmethod
    def _rz(alpha):
        return np.array([[np.exp(-1j * alpha / 2.0), 0.0],
                         [0.0, np.exp(1j * alpha / 2.0)]],
                        dtype=np.complex128)

    # ------------------------------------------------------------------
    # Target generation: buildTarget strategy (Supplementary Alg. 3)
    # ------------------------------------------------------------------
    def generate_target(self):
        """
        lOAD TARGET FROM FILE:
        np.load(f"random_states/random_target_{episode}.npy")
        """
        state_no = np.random.randint(0, 50000)  # Assuming 10k pre-generated states
        self.target_unitary = np.load(f"random_states/random_target_{state_no}.npy")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------
    def reset(self):
        self.step_counter = -1
        self.action_list = []
        self.last_gate_label = None

        dim = 2 ** self.num_qubits
        self.current_unitary = np.eye(dim, dtype=np.complex128)

        self.prev_fidelity = self.average_gate_fidelity()
        state = self._get_observation()
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def step(self, action):
        self.step_counter += 1
        self.action_list.append(action)

        gate_label = self.action_to_gate[action]

        # For the agent actions, you can still fix θ = π/128 as in the paper’s base.
        theta = np.pi / 128.0
        if gate_label == 'Rx_pos':
            gate_matrix = self._rx(+theta)
        elif gate_label == 'Rx_neg':
            gate_matrix = self._rx(-theta)
        elif gate_label == 'Ry_pos':
            gate_matrix = self._ry(+theta)
        elif gate_label == 'Ry_neg':
            gate_matrix = self._ry(-theta)
        elif gate_label == 'Rz_pos':
            gate_matrix = self._rz(+theta)
        elif gate_label == 'Rz_neg':
            gate_matrix = self._rz(-theta)
        else:
            raise ValueError(f"Unknown gate label: {gate_label}")

        self.current_unitary = gate_matrix @ self.current_unitary
        self.last_gate_label = gate_label

        next_state = self._get_observation()

        fidelity = self.average_gate_fidelity()
        op_err = self.operator_norm_error()

        if self.epsilon is not None:
            error_achieved = (op_err <= self.epsilon)
        else:
            error_achieved = (fidelity >= self.tolerance)

        max_steps_reached = self.step_counter >= (self.max_episode_length - 1)
        done = error_achieved or max_steps_reached

        if self.epsilon is not None:
            self.error = op_errr
        else:
            self.error = 1.0 - fidelity

        reward = self.compute_reward(fidelity, error_achieved)
        self.prev_fidelity = fidelity

        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)

        return next_state_t, reward_t, done

    # ------------------------------------------------------------------
    # Metrics, reward, logging (unchanged)r
    # ------------------------------------------------------------------
    def operator_norm_error(self):
        diff = self.target_unitary - self.current_unitary
        svals = np.linalg.svd(diff, compute_uv=False)
        return float(svals[0])

    def average_gate_fidelity(self):
        d = self.current_unitary.shape[0]
        trace = np.trace(self.target_unitary.conj().T @ self.current_unitary)
        fidelity = (np.abs(trace) ** 2 + d) / (d * (d + 1))
        return float(fidelity)

    def compute_reward(self, fidelity, error_achieved):
        if self.reward_type == 'dense':
            L = self.max_episode_length
            n = self.step_counter
            if error_achieved:
                reward = (L - n + 1)
            else:
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

    def make_circuit(self):
        qc = QuantumCircuit(self.num_qubits)
        for action in self.action_list:
            gate_label = self.action_to_gate[action]
            theta = np.pi / 128.0
            if gate_label == 'Rx_pos':
                qc.rx(+theta, 0)
            elif gate_label == 'Rx_neg':
                qc.rx(-theta, 0)
            elif gate_label == 'Ry_pos':
                qc.ry(+theta, 0)
            elif gate_label == 'Ry_neg':
                qc.ry(-theta, 0)
            elif gate_label == 'Rz_pos':
                qc.rz(+theta, 0)
            elif gate_label == 'Rz_neg':
                qc.rz(-theta, 0)
        return qc

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
            'error': 1.0 - fidelity
        }

if __name__ == "__main__":
    # Base config; tolerance will be overwritten in the loop
    base_config = {
        'env': {
            'num_qubits': 1,
            'max_episode_length': 300,
            'tolerance': 0.9999,   # placeholder; changed per run
            'reward_type': 'dense',
            'mode': 'paper',
            # optional: epsilon for operator norm stopping
            # 'epsilon': 1e-4,
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fidelities = [0.998, 0.9985, 0.999, 0.9995,
                  0.9996, 0.9997, 0.9998, 0.9999]
    num_episodes = 10000

    all_mean_lengths = []
    all_std_lengths = []
    all_solved_fracs = []

    # Loop over fidelity thresholds
    for F_target in fidelities:
        print(f"\n=== Random search for target AGF >= {F_target} ===")

        # Clone config and set tolerance
        config = copy.deepcopy(base_config)
        config['env']['tolerance'] = F_target

        # One env reused across episodes, but target will change every episode
        env = CircuitEnv(config, device)

        lengths = []
        solved_flags = []

        for ep in range(num_episodes):
            # Build a NEW random target for this episode
            env.generate_target()              # buildTarget: random N, random small rotations

            # Reset env with this target
            state = env.reset()

            done = False
            steps = 0
            while not done and steps < env.max_episode_length:
                a = np.random.randint(0, env.action_size)
                state, reward, done = env.step(a)
                steps += 1

            info = env.get_episode_info()
            lengths.append(info['sequence_length'])
            solved_flags.append(info['fidelity'] >= F_target)

        lengths = np.array(lengths, dtype=np.float64)
        solved_flags = np.array(solved_flags, dtype=bool)
        mean_len = lengths.mean()
        std_len = lengths.std()
        solved_frac = solved_flags.mean()

        all_mean_lengths.append(mean_len)
        all_std_lengths.append(std_len)
        all_solved_fracs.append(solved_frac)

        print(f"Episodes:          {num_episodes}")
        print(f"Mean length:       {mean_len:.2f}")
        print(f"Std length:        {std_len:.2f}")
        print(f"Solved fraction:   {solved_frac*100:.2f}%")