"""
Problem 3 - splitting M tasks across N agents using ADMM.

The catch is that standard sharing-ADMM assumes someone can see everyone's
variables at once to compute the average, which defeats the point of having
a communication graph in the first place. So instead of a central average,
each agent estimates the network average itself by gossiping with its
neighbors for a few rounds every iteration.
"""

from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

Path("figures").mkdir(exist_ok=True)
np.random.seed(11)

# --- setup ---
N = 5  # agents
M = 15  # tasks - has to be more than N per the problem

# connected graph again, same approach as problem 2
while True:
    G = nx.erdos_renyi_graph(N, 0.5, seed=np.random.randint(0, 10_000))
    if nx.is_connected(G):
        break

A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(axis=1)
max_deg = deg.max()
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j and A[i, j] == 1:
            W[i, j] = 1.0 / (max_deg + 1)
    W[i, i] = 1.0 - W[i, :].sum()

# random capacities that add up to M - reroll if anyone ends up with 0
b = np.random.multinomial(M, np.ones(N) / N)
while (b == 0).any():
    b = np.random.multinomial(M, np.ones(N) / N)
print("Agent capacities b_i:", b, "sum =", b.sum())


def run_decentralized_admm(C, rho=2.0, iters=150, consensus_rounds=8):
    """
    Runs ADMM where nobody has a global view - each agent only ever talks to
    its neighbors. consensus_rounds sets how many gossip rounds happen per
    ADMM iteration to approximate the true network average.
    """
    X = np.zeros((N, M))
    U = np.zeros((N, M))
    target = np.ones(M) / N

    for _ in range(iters):
        # every agent solves its own small QP using only what it can see locally
        X_next = np.zeros((N, M))
        for i in range(N):
            x_var = cp.Variable(M)
            resid = x_var - X[i, :] + (W @ X)[i, :] - target + U[i, :]
            objective = cp.Minimize(C[i, :] @ x_var + (rho / 2) * cp.sum_squares(resid))
            constraints = [cp.sum(x_var) == b[i], x_var >= 0, x_var <= 1]
            cp.Problem(objective, constraints).solve(solver=cp.OSQP, verbose=False)
            X_next[i, :] = x_var.value
        X = X_next

        # a few rounds of gossiping so everyone's estimate of the average gets closer
        Xbar_local = X.copy()
        for _ in range(consensus_rounds):
            Xbar_local = W @ Xbar_local

        # dual update, using each agent's own (approximate) average
        U = U + Xbar_local - target

    return X


# run 1 - just some random costs to start with
C1 = np.random.uniform(1, 10, (N, M))
X1 = run_decentralized_admm(C1)

# run 2 - same setup, but make agent 0 way cheaper and see what shifts
C2 = np.random.uniform(1, 10, (N, M))
C2[0, :] *= 0.25
X2 = run_decentralized_admm(C2)

for name, X in [("Run 1 (baseline costs)", X1), ("Run 2 (agent 0 cheap)", X2)]:
    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)
    print(
        f"{name}: row sums (should equal b_i) = {np.round(row_sums, 2)}, "
        f"max |col_sum-1| = {np.max(np.abs(col_sums - 1)):.4f}"
    )

# --- plot everything: the graph plus both allocation heatmaps side by side ---
fig = plt.figure(figsize=(15, 4.5))

ax0 = fig.add_subplot(131)
pos = nx.spring_layout(G, seed=2)
nx.draw(
    G,
    pos,
    ax=ax0,
    with_labels=True,
    node_color="#b7f0b0",
    node_size=500,
    font_weight="bold",
    edge_color="#555",
)
ax0.set_title(f"Agent graph $\\mathcal{{G}}$ (N={N}, connected)")

ax1 = fig.add_subplot(132)
im1 = ax1.matshow(np.round(X1, 2), cmap="Blues", vmin=0, vmax=1)
ax1.set_title("Allocation $X$ \u2014 Run 1 costs")
ax1.set_xlabel("Task $j$")
ax1.set_ylabel("Agent $i$")
ax1.xaxis.set_ticks_position("bottom")
fig.colorbar(im1, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(133)
im2 = ax2.matshow(np.round(X2, 2), cmap="Oranges", vmin=0, vmax=1)
ax2.set_title("Allocation $X$ \u2014 Run 2 (agent 0 cheap)")
ax2.set_xlabel("Task $j$")
ax2.set_ylabel("Agent $i$")
ax2.xaxis.set_ticks_position("bottom")
fig.colorbar(im2, ax=ax2, fraction=0.046)

fig.tight_layout()
fig.savefig("figures/p3_admm.png", dpi=200)
plt.close(fig)

print("figures saved")
print("\nRun 1 allocation (rounded):\n", np.round(X1, 2))
print("\nRun 2 allocation (rounded):\n", np.round(X2, 2))
