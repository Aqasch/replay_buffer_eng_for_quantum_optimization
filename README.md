
# Version 1: Reaply-Buffer Engineering for noise-aware quantum circuit optimization is on [arxiv](https://arxiv.org/abs/2604.21863)
### (The code is up to date but the camera ready paper will be available soon)
------------------
<h1 align="center">NeurIPS 2026</h1>

<p align="center">
  <img src="pics/model.jpg" alt="Sublime's custom image" width="1000" />
</p>


## Whats New?

### 🔁 **ReaPER+ (Annealed prioritized experience replay)**

- **Smoothly transitions** from TD-error-driven prioritization (PER, https://arxiv.org/abs/1511.05952) to reliability-aware sampling (ReaPER, https://iclr.cc/virtual/2026/poster/10008022) over the course of training.

<p align="center">
  <img src="pics/test.gif" alt="Sublime's custom image" width="300" />
</p>

- **Outperforms** fixed PER, ReaPER, HER, and PPO baselines on 1- and 2-qubit quantum compilation tasks, achieving up to 4× gains in sample efficiency and consistently discovering shorter circuits.

- **Validated beyond the quantum domain on the classical LunarLander-v3** benchmark, achieving a 9% AUC improvement over both PER and fixed ReaPER — confirming the annealing principle is domain-agnostic

### 🔀 **Lightweight replay-buffer transfer for noisy settings**

- **A weight-free buffer transfer scheme** that reuses noiseless trajectories to warm-start RL training in depolarizing-noise environments — without transferring network weights or reward relabeling.
<p align="center">
  <img src="pics/transfer.jpg" alt="Sublime's custom image" width="500" />
</p>

- **Reduces steps to chemical accuracy** by up to 85–90% and improves final energy error by up to 90% over from-scratch noisy baselines on 6-, 8-, and 12-qubit molecular tasks (such as BeH₂, H₂O).

- **Transfer advantage scales with system size:** at 12 qubits under combined depolarizing noise, buffer transfer reduces steps by 88.2% and achieves the highest composite transfer score of 51.0 across all benchmarks.

### ⚡ OptCRLQAS: Amortized curriculum learning

- Batches $m$ architectural edits before triggering a single quantum-classical evaluation, reducing wall-clock time per episode by up to 67.5% on 12-qubit H₂O without loss in accuracy.
<p align="center">
  <img src="pics/optcrlqas.jpg" alt="Sublime's custom image" width="300" />
</p>
- Cuts quantum circuit simulation time by up to 89% and classical optimization time by up to 85%.


### 📈 Improvement over nonRL baselines

| Problem    | Method                        | Min error (Ha)        | Total gates | CNOT |
|------------|-------------------------------|-----------------------|-------------|------|
| 5-Heisenberg | **OptCRLQAS + ReaPER+ (ours)** | **5.9 × 10⁻⁴**   | **41**      | NA   |
|            | [DQAS](https://iopscience.iop.org/article/10.1088/2058-9565/ac87cd/meta)                     | 1.1 × 10⁻¹            | 35          | NA   |
|            | [GQAS](https://www.sciencedirect.com/science/article/pii/S0893608024004325)                     | 7.1 × 10⁻⁴            | 35          | NA   |
|            | [TF-QAS](https://ojs.aaai.org/index.php/AAAI/article/view/29135)                   | 1.2 × 10⁻³            | 35          | NA   |
| 6-BEH₂     | **OptCRLQAS + ReaPER+ (ours)** | **5.8 × 10⁻⁵**   | **54**      | **12** |
|            | [TF-QAS](https://ojs.aaai.org/index.php/AAAI/article/view/29135)                   | 1.8 × 10⁻³            | 57          | NA   |
|            | [SA-QAS](https://proceedings.mlr.press/v202/lu23f.html)                   | 5.6 × 10⁻³            | 73          | 45   |
| 8-H₂O      | **OptCRLQAS + ReaPER+ (ours)** | **1.2 × 10⁻⁴**   | **134**     | **52** |
|            | [quantumDARTS](https://proceedings.mlr.press/v202/wu23v.html)             | 1.7 × 10⁻⁴            | 219         | 68   |
|            | [SA-QAS](https://proceedings.mlr.press/v202/lu23f.html)                   | 2.6 × 10⁻³            | 95          | 69   |



## Running code
```
conda create -n {name_your_environment} python=3.10
conda activate {name_your_environment}
pip install -r requirements.txt
```

## Quantum compiling

**(Kindly use `-h` at the end of each python script to see all available options for setting arguments)**

For the 1-qubit compiling with action space consisting of $\texttt{RX}, \texttt{RY}, \texttt{RZ}$ with $\pm\pi/128$ angle simple run:

```
python compiling/main_1q_small_rot.py --replay CHOOSE_BUFFER
```

CHOOSE_BUFFER can have the following entries: \{'per', 'reaper','her' or 'reaper_anneal'\}$. So for **ReaPER+** you run:
```
python compiling/main_1q_small_rot.py --replay reaper_anneal
```

For 1-qubit [HRC gateset](https://arxiv.org/abs/quant-ph/0111031) simply run (say with **ReaPER**):
```
python compiling/main_1q_hrc.py --replay reaper
```
Finally for 2-qubit compiling just run (say with **PER**):
```
python compiling/main_2q.py --replay per
```

## Quantum architecture search (QAS)

The RL-algorithm is isnpired by [CRLQAS](https://openreview.net/forum?id=rINBD8jPoP) but with `m` **step ammortization** and **replay buffer transfer**. Further discussed in **Whats New?** below. For the sake of reproducing, we share instructions to run code for 6- and 8-qubit QAS. The same instructions follow for 10- and 12-qubit.

### Engineering replay buffer
The engineering of replay buffer contains corresponds to how a specific quantum architecture search with (i) uniform random sample (say for 8-qubit QAS):
```
python main_ammortized.py --seed 42 --config 8q --experiment_name "uniform_sample/"
```

(ii) [prioritized experience replay (PER)](https://arxiv.org/abs/1511.05952) (with 6-qubit QAS):
```
python main_ammortized.py --seed 42 --config 6q --experiment_name "PER/"
```

(iii) [Realibility-adjusted PER (ReaPER)](https://iclr.cc/virtual/2026/poster/10008022) (with 8-qubit QAS and $\omega=0.6$)
```
python main_ammortized.py --seed 42 --config 8q_omega0p6 --experiment_name "ReaPER/"
```

The $\omega$ can take values `0p2`, `0p4`, `0p6`, `0p8`.

(iv) and **our introduced ReaPER+** which anneals from PER to ReaPER as the traning progresses (with 6-qubit QAS).
```
python main_ammortized.py --seed 42 --config 6q --experiment_name "ReaPER_anneal/"
```


### $m$-step ammortized QAS

To run the $m$-step ammortized quantum architecture search use the following commend:
```
python main_ammortized.py --seed {SEED} --config {N}q_{m}step --experiment_name "ammortized/"
```

where *SEED* can be any integer value. *N* is the number of qubit and *m* is the ammortization step. For an example, to run the 6-qubit ground state finding problem with 10-step ammortization you simply copy the following:
```
python main_ammortized.py --seed 42 --config 6q_10step --experiment_name "ammortized/"
```

### Buffer transfer

We conduct the transfer of buffer from a `noiseless` scenario to `realistic noisy` cases. To initiate the transfer of buffer you need to run the `noiseless` case and save the replay buffer. If saved then you can load it in a `realistic noisy` scenario and initiate the transfer. First run the noiseless case:
```
python main_ammortized.py --seed 42 --config 6q_wo_t --experiment_name "buffer_transfer/"
```

In `configs/buffer_transfer/` you can find the configuration files. These files can be modified to run different instances of the lightweight transfer. For example if you want to transfer the buffer from `1000`th in `noiseless` case to `noise (2-qubit depolarizing with 0.001 noise strength)` you simple need to make `buffer_ep = 1000` in the specific configuration and run:

```
python main_ammortized.py --seed 42 --config 6q_w_t_0p001 --experiment_name "buffer_transfer/"
```
