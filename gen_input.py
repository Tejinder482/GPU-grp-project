"""
Makes an input file for ./nbody and ./barnes_hut.

    python3 gen_input.py N out.txt [--steps 10] [--dt 1e-3] [--G 1] [--eps 1e-4] [--seed 1]

Particles are spread uniformly inside a sphere of radius 1 with random
velocities. Masses are random in [0.5, 1.5] and then scaled so the total mass
is 1, so with G = 1 everything is in standard N-body units.
"""
import argparse

import numpy as np


def make_cluster(n, rng):
    # uniform inside the unit sphere: random direction, radius ~ u^(1/3)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = rng.random(n) ** (1.0 / 3.0)
    pos = d * r[:, None]

    m = rng.uniform(0.5, 1.5, n)
    m /= m.sum()

    # velocity spread a bit below the virial value (~0.45 per component for
    # this sphere), so the cluster slowly contracts instead of flying apart
    vel = rng.normal(scale=0.3, size=(n, 3))
    vel -= (m[:, None] * vel).sum(axis=0) / m.sum()  # no net drift
    return pos, vel, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    pos, vel, m = make_cluster(a.n, np.random.default_rng(a.seed))
    data = np.column_stack([pos, vel, m])
    with open(a.out, "w") as f:
        f.write("# N steps dt G eps, then x y z vx vy vz m per particle\n")
        f.write(f"{a.n} {a.steps} {a.dt:g} {a.G:g} {a.eps:g}\n")
        np.savetxt(f, data, fmt="%.8e")


if __name__ == "__main__":
    main()
