import os
import numpy as np
from tqdm import tqdm

SAVE_DIR = "random_states"
NUM_TARGETS = 50000
SEED = 42

# Choose ONE:
# N_MIN, N_MAX = 5, 105        # if you interpret Algorithm 3 literally as 5..105
N_MIN, N_MAX = 5, 5000       # if you interpret the PDF text as 5..10^5

ALPHA = np.pi / 128


def rx(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=np.complex128)


def ry(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -s],
        [s, c]
    ], dtype=np.complex128)


def rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)


def build_target(rng, n_min=N_MIN, n_max=N_MAX, alpha=ALPHA):
    gates = [
        rx(+alpha),
        ry(+alpha),
        rz(+alpha),
        rx(-alpha),
        ry(-alpha),
        rz(-alpha),
    ]

    # inclusive upper bound
    N = rng.integers(n_min, n_max + 1)

    U = np.eye(2, dtype=np.complex128)
    seq = []

    for _ in range(N):
        idx = rng.integers(0, len(gates))
        U = gates[idx] @ U
        seq.append(idx)

    return U, np.array(seq, dtype=np.int8)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    lengths = []

    for i in tqdm(range(NUM_TARGETS)):
        U, seq = build_target(rng)
        np.save(os.path.join(SAVE_DIR, f"random_target_{i}.npy"), U)
        # optional: also save the sequence for debugging/reproducibility
        np.save(os.path.join(SAVE_DIR, f"random_target_{i}_seq.npy"), seq)
        lengths.append(len(seq))

    lengths = np.array(lengths)
    np.save(os.path.join(SAVE_DIR, "target_lengths.npy"), lengths)

    print("Saved targets:", NUM_TARGETS)
    print("Min length:", lengths.min())
    print("Max length:", lengths.max())
    print("Mean length:", lengths.mean())


if __name__ == "__main__":
    main()