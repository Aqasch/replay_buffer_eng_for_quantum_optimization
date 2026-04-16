import torch
from qulacs.gate import CNOT, RX, RY, RZ
from utils import *
from sys import stdout
import scipy
import VQE as vc
import os
import numpy as np
import copy
import curricula
from collections import Counter
try:
    from qulacs import QuantumStateGpu as QuantumState
except ImportError:
    from qulacs import QuantumState

from qulacs import ParametricQuantumCircuit, Observable
import copy


class CircuitEnv():

    def __init__(self, conf, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        self.random_halt = int(conf['env']['rand_halt'])

        self.n_shots = conf['env']['n_shots']
        noise_models = int(conf['env']['noise_models'])
        if noise_models:
            noise_values = conf['env']['noise_values'].strip('[] ').split(',')
            self.noise_values = [float(x.strip()) for x in noise_values]
            if len(self.noise_values) > 1:
                noise_model_label = f'2-qubit ({self.noise_values[0]}) & 1-qubit ({self.noise_values[1]})'
            else:
                noise_model_label = f'2-qubit ({self.noise_values[0]})'
        else:
            noise_model_label = 'Noiseless'
            self.noise_values = []

        self.noise_models = 'depolarizing'


        self.mol = conf['problem']['ham_type']

        self.phys_noise = False
        self.err_mitig = conf['env']['err_mitig']

        self.ham_mapping = conf['problem']['mapping']
        self.geometry = conf['problem']['geometry'].replace(" ", "_")

        self.fake_min_energy = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else None
        self.fn_type = conf['env']['fn_type']

        if "cnot_rwd_weight" in conf['env'].keys():
            self.cnot_rwd_weight = conf['env']['cnot_rwd_weight']
        else:
            self.cnot_rwd_weight = 1.

        self.noise_flag = True
        self.state_with_angles = conf['agent']['angles']
        self.current_number_of_cnots = 0

        self.curriculum_dict = {}
        __ham = np.load(f"mol_data/{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        print('=============================')
        print(f" The training happening for {self.mol}:\n -> with {self.num_qubits} qubit\n -> geometry: {self.geometry}\n -> Fermion-to-qubit mapping: {self.ham_mapping}\n")
        print(f"(*) Ammortization step: {conf['env']['energy_interval']}\n")

        if conf['agent']['init_buffer'] == 0:
            print('(*) WITHOUT any buffer transfer')
        else:
            print(f'(*) WITH {noise_model_label} {self.noise_models} noise buffer transfer:\n')
            e = conf['agent']['buffer_ep']
            epsilon_warm = conf['agent']['epsilon_warm']
            print(f'(*) Transfering buffer upto episode {e} from noiseless run\n')
            print(f'(*) Epsilon is warmstarted with: {epsilon_warm}\n')

        print('=============================')

        _, _, eigvals, energy_shift = __ham['hamiltonian'], __ham['weights'], __ham['eigvals'], __ham['energy_shift']

        min_eig = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else min(eigvals) + energy_shift

        self.hamiltonian, self.weights, eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'], __ham['eigvals'], __ham['energy_shift']

        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.max_eig = max(eigvals) + self.energy_shift

        self.curriculum_dict[self.geometry[-3:]] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_energy=min_eig)

        self.device = device
        self.ket = QuantumState(self.num_qubits)
        self.done_threshold = conf['env']['accept_err']

        stdout.flush()
        self.state_size = self.num_layers * self.num_qubits * (self.num_qubits + 3 + 3)
        self.step_counter = -1
        self.prev_energy = None
        self.moments = [0] * self.num_qubits
        self.illegal_actions = [[]] * self.num_qubits
        self.energy = 0
        self.opt_alg_save = 0

        self.action_size = (self.num_qubits * (self.num_qubits + 2))
        self.previous_action = [0, 0, 0, 0]

        eigvals, weights, pauli_strings = __ham['eigvals'], __ham['weights'], __ham['paulis']
        self.hamiltonian_op, self.qulacs_energy_shift = self.qulacs_hamiltonian_form(weights, pauli_strings)

        if 'non_local_opt' in conf.keys():
            self.global_iters = conf['non_local_opt']['global_iters']
            self.optim_method = conf['non_local_opt']["method"]
            self.optim_alg = conf['non_local_opt']['optim_alg']

            if 'a' in conf['non_local_opt'].keys():
                self.options = {
                    'a': conf['non_local_opt']["a"],
                    'alpha': conf['non_local_opt']["alpha"],
                    'c': conf['non_local_opt']["c"],
                    'gamma': conf['non_local_opt']["gamma"],
                    'beta_1': conf['non_local_opt']["beta_1"],
                    'beta_2': conf['non_local_opt']["beta_2"]
                }

            if 'lamda' in conf['non_local_opt'].keys():
                self.options['lamda'] = conf['non_local_opt']["lamda"]

            if 'maxfev' in conf['non_local_opt'].keys():
                self.maxfev = {}
                self.maxfev['maxfev'] = int(conf['non_local_opt']["maxfev"])

            if 'maxfev1' in conf['non_local_opt'].keys():
                self.maxfevs = {}
                self.maxfevs['maxfev1'] = int(conf['non_local_opt']["maxfev1"])
                self.maxfevs['maxfev2'] = int(conf['non_local_opt']["maxfev2"])
                self.maxfevs['maxfev3'] = int(conf['non_local_opt']["maxfev3"])
        else:
            self.global_iters = 0
            self.optim_method = None
        
        # print(self.global_iters, self.optim_method)

        # New: intervals for energy evaluation and inner optimizer
        self.energy_interval = conf['env'].get('energy_interval', 1)
        self.opt_step_interval = conf['env'].get('opt_step_interval', 1)

        # Episode counter (incremented in reset)
        self.episode_counter = 0

        # Last energy value used between evaluations
        self.prev_energy = self.max_eig

    def step(self, action, train_flag=True):

        next_state = self.state.clone()
        self.step_counter += 1

        ctrl = action[0]
        targ = (action[0] + action[1]) % self.num_qubits
        rot_qubit = action[2]
        rot_axis = action[3]

        self.action = action

        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[rot_qubit]
        elif ctrl < self.num_qubits:
            gate_tensor = max(self.moments[ctrl], self.moments[targ])
        else:
            gate_tensor = 0  # safe default; should not be used if no gate

        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits+rot_axis-1][rot_qubit] = 1

        if rot_qubit < self.num_qubits:
            self.moments[rot_qubit] += 1
        elif ctrl < self.num_qubits:
            max_of_two_moments = max(self.moments[ctrl], self.moments[targ])
            self.moments[ctrl] = max_of_two_moments + 1
            self.moments[targ] = max_of_two_moments + 1

        self.current_action = action
        self.illegal_action_new()

        # Decide if this episode is allowed to run the inner optimizer
        # run_inner_opt = (self.episode_counter % self.optimizer_interval == 0)
        run_inner_opt = (self.step_counter % self.opt_step_interval == 0)

        if run_inner_opt and self.optim_method in ["scipy_each_step"]:
            thetas, nfev, opt_ang = self.scipy_optim(self.optim_alg)
            for i in range(self.num_layers):
                for j in range(3):
                    next_state[i][self.num_qubits+3+j, :] = thetas[i][j, :]
        self.state = next_state.clone()

        # Evaluate energy only every energy_interval steps or at episode end
        layers_done = self.step_counter == (self.num_layers - 1)
        should_eval = (self.step_counter % self.energy_interval == 0) or layers_done

        if should_eval:
            energy, energy_noiseless = self.get_energy()
        else:
            energy = self.prev_energy
            energy_noiseless = self.prev_energy

        if self.noise_flag is False:
            energy = energy_noiseless

        self.energy = energy

        if energy < self.curriculum.lowest_energy and train_flag and should_eval:
            self.curriculum.lowest_energy = copy.copy(energy)

        self.error = float(abs(self.min_eig - energy))
        self.error_noiseless = float(abs(self.min_eig - energy_noiseless))

        # print(self.error, self.done_threshold)

        rwd = self.reward_fn(energy)
        self.prev_energy = float(energy)

        energy_done = int(self.error < self.done_threshold)
        done = int(energy_done or layers_done)

        self.previous_action = copy.deepcopy(action)

        if self.random_halt:
            if self.step_counter == self.halting_step:
                done = 1
        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(self.curriculum)

        if self.state_with_angles:
            return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
        else:
            next_state = next_state[:, :self.num_qubits+3]
            return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done

    def reset(self):

        state = torch.zeros((self.num_layers, self.num_qubits+3+3, self.num_qubits))
        self.state = state

        if self.random_halt:
            statistics_generated = np.clip(
                np.random.negative_binomial(n=70, p=0.573, size=1), 25, 70
            )[0]
            self.halting_step = statistics_generated

        # # Random angle initialization in [-init_scale, init_scale]
        # init_scale = np.pi  # or smaller like np.pi/4
        # # angle block is shape [num_layers, 3, num_qubits]
        # rand_thetas = (2 * init_scale) * torch.rand(
        #     (self.num_layers, 3, self.num_qubits)
        # ) - init_scale
        # self.state[:, self.num_qubits+3:] = rand_thetas
        # state = self.state

        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits] * 4
        self.illegal_actions = [[]] * self.num_qubits

        self.make_circuit(state)
        self.step_counter = -1

        self.moments = [0] * self.num_qubits
        self.current_bond_distance = self.geometry[-3:]
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        self.geometry = self.geometry[:-3] + str(self.current_bond_distance)
        __ham = np.load(f"mol_data/{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        self.weights, eigvals, self.energy_shift, self.pauli_strings = __ham['weights'], __ham['eigvals'], __ham['energy_shift'], __ham['paulis']
        self.hamiltonian_op, self.qulacs_energy_shift = self.qulacs_hamiltonian_form(self.weights, self.pauli_strings)

        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.max_eig = max(eigvals) + self.energy_shift

        # initial exact energy for this episode
        self.prev_energy = self.get_energy(state)[1]

        # count episode
        self.episode_counter += 1

        if self.state_with_angles:
            return state.reshape(-1).to(self.device)
        else:
            state = state[:, :self.num_qubits+3]
            return state.reshape(-1).to(self.device)

    def make_circuit(self, thetas=None):
        state = self.state.clone()
        if thetas is None:
            thetas = state[:, self.num_qubits+3:]

        circuit = ParametricQuantumCircuit(self.num_qubits)

        for i in range(self.num_layers):

            cnot_pos = np.where(state[i][0:self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circuit.add_gate(CNOT(ctrl[r], targ[r]))

            rot_pos = np.where(state[i][self.num_qubits: self.num_qubits+3] == 1)

            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]

            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        circuit.add_parametric_RX_gate(rot_qubit, thetas[i][0][rot_qubit])
                    elif r == 1:
                        circuit.add_parametric_RY_gate(rot_qubit, thetas[i][1][rot_qubit])
                    elif r == 2:
                        circuit.add_parametric_RZ_gate(rot_qubit, thetas[i][2][rot_qubit])
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r > 2

        return circuit

    def qulacs_hamiltonian_form(self, weights, paulis_string):
        obs = Observable(self.num_qubits)
        energy_shift = 0.0

        for coeff_scalar, pstr in zip(weights, paulis_string):
            if pstr == "I" * self.num_qubits:
                energy_shift += coeff_scalar
                continue

            terms = []
            for qubit_idx, pauli in enumerate(pstr):
                if pauli != "I":
                    terms.append(f"{pauli} {qubit_idx}")

            qulacs_compatible_form = " ".join(terms) if terms else "I 0"
            obs.add_operator(coeff_scalar, qulacs_compatible_form)

        return obs, energy_shift

    def get_energy(self, thetas=None):
        qulacs_inst = vc.Parametric_Circuit(
            n_qubits=self.num_qubits,
            noise_models=self.noise_models,
            noise_values=self.noise_values
        )
        circ = qulacs_inst.construct_ansatz(self.state)
        expval_noiseless = vc.get_exp_val(
            self.num_qubits, circ, self.hamiltonian_op, self.qulacs_energy_shift
        )
        energy = expval_noiseless
        return energy, energy

    def scipy_optim(self, method, which_angles=[]):
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:, self.num_qubits: self.num_qubits+3] == 1).nonzero(as_tuple=True)
        angles = thetas[rot_pos]

        qulacs_inst = vc.Parametric_Circuit(
            n_qubits=self.num_qubits,
            noise_models=self.noise_models,
            noise_values=self.noise_values
        )
        qulacs_circuit = qulacs_inst.construct_ansatz(state)

        x0 = np.asarray(angles.cpu().detach())

        def cost(x):
            return vc.get_energy_qulacs(
                x, observable=self.hamiltonian_op,
                weights=self.weights, circuit=qulacs_circuit,
                n_qubits=self.num_qubits, qulacs_energy_shift=self.energy_shift,
                n_shots=int(self.n_shots), phys_noise=self.phys_noise,
                which_angles=[]
            )

        if x0.size == 0:
            return thetas, 0, x0
        else:
            result_min_qulacs = scipy.optimize.minimize(
                cost, x0=x0, method=method,
                options={'maxiter': self.global_iters}
            )
            thetas = state[:, self.num_qubits+3:]
            thetas[rot_pos] = torch.tensor(result_min_qulacs['x'], dtype=torch.float)
            return thetas, result_min_qulacs['nfev'], result_min_qulacs['x']

    def spsa_optim(self, n_steps=3, alpha=0.602, gamma=0.101,
                   a0=0.1, c0=0.1):
        """
        Simple SPSA optimizer on the current parametric angles.

        - n_steps: how many SPSA iterations per call
        - alpha, gamma: standard SPSA exponents
        - a0, c0: base learning-rate and perturbation scales
        """
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:, self.num_qubits: self.num_qubits+3] == 1).nonzero(as_tuple=True)
        angles = thetas[rot_pos]  # 1D tensor of active parameters

        # If no trainable angles, do nothing
        if angles.numel() == 0:
            return thetas, 0, angles

        # Work in numpy for convenience
        theta = np.asarray(angles.cpu().detach(), dtype=float)

        def set_thetas_from_flat(vec):
            # write back flattened vector into thetas using rot_pos
            nonlocal thetas
            thetas[rot_pos] = torch.tensor(vec, dtype=torch.float32)

        def cost_from_flat(vec):
            set_thetas_from_flat(vec)
            # temporarily update self.state's angle block
            self.state[:, self.num_qubits+3:] = thetas
            E, _ = self.get_energy()
            return float(E)

        k = 0
        for step in range(n_steps):
            k += 1
            a_k = a0 / (k ** alpha)
            c_k = c0 / (k ** gamma)

            # Rademacher perturbation
            delta = 2 * (np.random.randint(0, 2, size=theta.shape) - 0.5)
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta

            # Two energy evaluations
            loss_plus = cost_from_flat(theta_plus)
            loss_minus = cost_from_flat(theta_minus)

            # SPSA gradient estimate
            g_hat = (loss_plus - loss_minus) / (2.0 * c_k * delta)

            # Parameter update
            theta = theta - a_k * g_hat

        # Write back final theta into thetas and state
        set_thetas_from_flat(theta)
        self.state[:, self.num_qubits+3:] = thetas

        # nfev = 2 per step
        nfev = 2 * n_steps
        return thetas, nfev, theta


    def reward_fn(self, energy):
        
        
        if self.fn_type == "staircase":
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "two_step":
            return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
        elif self.fn_type == "two_step_end":

            max_depth = self.step_counter == (self.num_layers - 1)
            if ((self.error < self.done_threshold) or max_depth):
                return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
            else:
                return 0.0
        elif self.fn_type == "naive":
            return 0. + 1.*(self.error < self.done_threshold)
        elif self.fn_type == "incremental":
            return (self.prev_energy - energy)/abs(self.prev_energy - self.min_eig)
        elif self.fn_type == "incremental_clipped":
            return np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
        elif self.fn_type == "nive_fives":

            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = 0.
            return rwd
        
        elif self.fn_type == "incremental_with_fixed_ends":
            

            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            return rwd
        
        elif self.fn_type == "log":
            return -np.log(1-(energy/self.min_eig))
        
        elif self.fn_type == "log_to_ground":
            
            return -np.log(self.error)
        
        elif self.fn_type == "log_to_threshold":
            if self.error < self.done_threshold + 1e-5:
                rwd = 11
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_0_end":
            rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_50_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 50
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_100_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 100
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_500_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 100
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
	
        elif self.fn_type == "log_to_threshold_500_end":
                if self.error < self.done_threshold + 1e-5:
                    rwd = 500
                else:
                    rwd = -np.log(abs(self.error - self.done_threshold))
                return rwd
        
        elif self.fn_type == "log_to_threshold_1000_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 1000
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_non_repeat_energy":
            if self.error < self.done_threshold + 1e-5:
                rwd = 30
            elif np.abs(self.energy-self.prev_energy) <= 1e-3:
                rwd = -30
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_no_repeat_actions":
            if self.current_action == self.previous_action:
                return -1 
            elif self.error < self.done_threshold + 1e-5:
                rwd = 20
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_neg_punish":
            return -np.log(1-(energy/self.min_eig)) - 5
        
        elif self.fn_type == "end_energy":

            max_depth = self.step_counter == (self.num_layers - 1)
            
            if ((self.error < self.done_threshold) or max_depth):
                rwd = (self.max_eig - energy) / (abs(self.min_eig) + abs(self.max_eig))
            else:
                rwd = 0.0

        elif self.fn_type == "hybrid_reward":
            path = 'threshold_crossed.npy'
            if os.path.exists(path):
                
                threshold_pass_info = np.load(path)
                if threshold_pass_info > 8:

                    max_depth = self.step_counter == (self.num_layers - 1)
                    if (self.error < self.done_threshold):
                        rwd = 5.
                    elif max_depth:
                        rwd = -5.
                    else:
                        rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
                    return rwd
                else:
                    if self.error < self.done_threshold + 1e-5:
                        rwd = 11
                    else:
                        rwd = -np.log(abs(self.error - self.done_threshold))
                    return rwd
            else:
                np.save('threshold_crossed.npy', 0)
        
        elif self.fn_type == 'negative_above_chem_acc':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = 1000*(self.done_threshold/self.error)
            return rwd
        
        elif self.fn_type == 'negative_above_chem_acc_non_increment':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = self.done_threshold/self.error
            return rwd
        
        elif self.fn_type == 'negative_above_chem_acc_slight_increment':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = 100*(self.done_threshold/self.error)
            return rwd


        elif self.fn_type == "cnot_reduce":

            max_depth = self.step_counter == (self.num_layers - 1)
            
            
            if (self.error < self.done_threshold):
                rwd = self.num_layers - self.cnot_rwd_weight*self.current_number_of_cnots
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            return 

        
    def illegal_action_new(self):
        action = self.current_action
        illegal_action = self.illegal_actions
        ctrl, targ = action[0], (action[0] + action[1]) % self.num_qubits
        rot_qubit, rot_axis = action[2], action[3]

        if ctrl < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[2] == self.num_qubits:
                        
                            if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[0] or targ == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break                          
            else:
                illegal_action[0] = action

                            
        if rot_qubit < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[0] == self.num_qubits:
                            
                            if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            elif rot_qubit != ill_ac[2]:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if rot_qubit == ill_ac[0]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                                        
                            elif rot_qubit == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break 
            else:
                illegal_action[0] = action
        
        for indx in range(self.num_qubits):
            for jndx in range(indx+1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx +1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break
        
        for indx in range(self.num_qubits-1):
            if len(illegal_action[indx])==0:
                illegal_action[indx] = illegal_action[indx+1]
                illegal_action[indx+1] = []
        
        illegal_action_decode = []
        for key, contain in dictionary_of_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        return illegal_action_decode




if __name__ == "__main__":
    pass
