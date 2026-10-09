"""
Problem 2 - tracking the intruder with a swarm of drones.

Each drone only gets a noisy reading of where the target is relative to its
own known position, so on its own no single drone can pin the target down.
They run distributed gradient descent over the network so the swarm fuses
its measurements into one shared estimate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for the 3d projection)

Path("figures").mkdir(exist_ok=True)
np.random.seed(42)

# --- setup ---
N = 8  # number of drones, staying under 20 as the problem asks
T_max = 60
p_prob = 0.45

# keep drawing random graphs until one of them is connected
while True:
    G = nx.erdos_renyi_graph(N, p_prob, seed=np.random.randint(0, 10_000))
    if nx.is_connected(G):
        break

A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(axis=1)
max_deg = deg.max()

# Metropolis weights - gives a doubly stochastic W so the averaging step converges
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j and A[i, j] == 1:
            W[i, j] = 1.0 / (max_deg + 1)
    W[i, i] = 1.0 - W[i, :].sum()

# fixed problem statistics (drawn once, not re-rolled every time step)
z0_bar = np.array([2.0, -1.0, 3.0])
Sigma_0 = np.eye(3) * 0.6
Sigma_w = np.eye(3) * 0.05  # how much the target moves per step
x_sensors = np.random.uniform(-8, 8, (N, 3))  # drones stay put at these locations
Sigma_v = [np.eye(3) * s for s in np.random.uniform(0.15, 0.6, N)]  # each drone's noise level
inv_sigma_v = [np.linalg.inv(Sv) for Sv in Sigma_v]

# simulate the target actually wandering around (random walk)
z_true = np.zeros((T_max, 3))
z_true[0] = np.random.multivariate_normal(z0_bar, Sigma_0)
for t in range(1, T_max):
    w = np.random.multivariate_normal(np.zeros(3), Sigma_w)
    z_true[t] = z_true[t - 1] + w

# --- distributed gradient descent tracking ---
alpha = 0.04  # had to bring this down a lot - anything much bigger diverges
K_iters = 20  # inner iterations per time step before moving on
z_est = np.zeros((T_max, N, 3))
z_est[0] = np.tile(z0_bar, (N, 1))

for t in range(T_max):
    # every drone takes one noisy measurement at this time step
    y = np.zeros((N, 3))
    for i in range(N):
        v = np.random.multivariate_normal(np.zeros(3), Sigma_v[i])
        y[i] = x_sensors[i] - z_true[t] + v

    z_k = z_est[t - 1].copy() if t > 0 else z_est[0].copy()
    for _ in range(K_iters):
        consensus = W @ z_k
        grad = np.array([inv_sigma_v[i] @ (z_k[i] - (x_sensors[i] - y[i])) for i in range(N)])
        z_k = consensus - alpha * grad
    z_est[t] = z_k

z_est_mean = z_est.mean(axis=1)
error = np.linalg.norm(z_est_mean - z_true, axis=1)
print(f"Mean steady-state ||e(t)|| over last 20 steps: {error[-20:].mean():.4f}")

# --- plot 1: just the graph, to show it's connected ---
fig_g, ax_g = plt.subplots(figsize=(4, 4))
pos = nx.spring_layout(G, seed=1)
nx.draw(
    G,
    pos,
    ax=ax_g,
    with_labels=True,
    node_color="#ffd28a",
    node_size=420,
    font_weight="bold",
    edge_color="#555",
)
ax_g.set_title(f"Drone communication graph $\\mathcal{{G}}$ (N={N}, p={p_prob})")
fig_g.tight_layout()
fig_g.savefig("figures/p2_graph.png", dpi=200)
plt.close(fig_g)

# --- plot 2: true path vs. what the swarm thinks is happening, plus the error ---
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2], "g-", lw=2, label="True intruder $z(t)$")
ax1.plot(
    z_est_mean[:, 0],
    z_est_mean[:, 1],
    z_est_mean[:, 2],
    "r--",
    lw=2,
    label="DGD estimate $\\hat{z}(t)$",
)
ax1.scatter(
    x_sensors[:, 0], x_sensors[:, 1], x_sensors[:, 2], c="b", marker="^", s=60, label="Drones"
)
ax1.scatter(*z_true[0], c="k", marker="o", s=50)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.set_title("Intruder tracking in $\\mathbb{R}^3$")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(122)
ax2.plot(range(T_max), error, "k-", lw=2)
ax2.set_xlabel("Time step $t$")
ax2.set_ylabel(r"Estimation error $\|e(t)\|$")
ax2.set_title("$\\|\\hat{z}(t)-z(t)\\|$ over time")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/p2_tracking.png", dpi=200)
plt.close(fig)

print("done - figures saved")
