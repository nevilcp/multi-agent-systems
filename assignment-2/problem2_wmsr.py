"""
Problem 2.1 - W-MSR resilient consensus on a 3-robust graph.

Eight agents (nodes 1-8 in the figure are indices 0-7 here) run W-MSR with
F = 1. Node 8 is the adversary and keeps broadcasting a fixed extreme value;
every benign node trims up to F larger and F smaller neighbour values before
averaging, so the adversary should never influence the consensus.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

Path("figures").mkdir(exist_ok=True)

# adjacency list for the 3-robust graph (0-indexed; nodes 1-8 in the figure -> 0-7 here)
adj = {
    0: [1, 2, 4, 6],  # node 1
    1: [0, 3, 4, 5],  # node 2
    2: [0, 3, 4, 6],  # node 3
    3: [1, 2, 5, 7],  # node 4
    4: [5, 6, 7, 0, 1, 2],  # node 5
    5: [4, 6, 7, 1, 3],  # node 6
    6: [4, 5, 7, 0, 2],  # node 7
    7: [4, 5, 6, 3],  # node 8
}

N = 8
F = 1  # robust against F = 1 local adversary
iters = 20
adversary = 7  # node 8 (index 7) is the malicious agent

np.random.seed(42)
x = np.random.uniform(0, 10, N)
x[adversary] = 100.0  # the adversary starts (and stays) at an extreme value

# the topology itself: outer ring = nodes 1-4, inner ring = nodes 5-8 (as in the figure)
G = nx.Graph([(i, j) for i, nbrs in adj.items() for j in nbrs])
# (the inner ring is rotated by 45 degrees so that no edge hides behind another one)
angles = np.arange(4) * np.pi / 2
pos = {i: (2.3 * np.cos(a + np.pi / 4), 2.3 * np.sin(a + np.pi / 4)) for i, a in enumerate(angles)}
pos |= {4 + i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
colors = ["#ff9d9d" if i == adversary else "#a8d0ff" for i in G.nodes]
plt.figure(figsize=(4.5, 4.5))
nx.draw(
    G,
    pos,
    labels={i: i + 1 for i in G.nodes},
    node_color=colors,
    node_size=420,
    font_weight="bold",
    edge_color="#555555",
)
plt.title("3-robust graph (adversary: node 8)")
plt.savefig("figures/p2_graph.png", dpi=200, bbox_inches="tight")
plt.close()

history = [x.copy()]

for _ in range(iters):
    x_next = np.zeros(N)
    for i in range(N):
        if i == adversary:
            x_next[i] = 100.0  # the adversary keeps its malicious extreme value
            continue

        neighbor_vals = sorted(x[j] for j in adj[i])

        # W-MSR filtering: drop up to F strictly larger and F strictly smaller values
        larger = [v for v in neighbor_vals if v > x[i]]
        smaller = [v for v in neighbor_vals if v < x[i]]
        removed_larger = larger[-F:] if larger else []
        removed_smaller = smaller[:F] if smaller else []

        kept = []
        for v in neighbor_vals:
            if v in removed_larger:
                removed_larger.remove(v)
            elif v in removed_smaller:
                removed_smaller.remove(v)
            else:
                kept.append(v)

        kept.append(x[i])  # a node always keeps its own value
        x_next[i] = sum(kept) / len(kept)

    x = x_next.copy()
    history.append(x.copy())

history = np.array(history)

benign = [i for i in range(N) if i != adversary]
print("Benign final values:", np.round(history[-1, benign], 4))
print(
    f"Benign spread at k={iters}: {np.ptp(history[-1, benign]):.2e} | "
    f"initial benign range: [{history[0, benign].min():.3f}, {history[0, benign].max():.3f}]"
)

plt.figure(figsize=(8, 5))
for i in range(N):
    if i == adversary:
        plt.plot(history[:, i], "r--", linewidth=2, label=f"Adversary (Node {i + 1})")
    else:
        plt.plot(history[:, i], label=f"Benign Node {i + 1}")

plt.title("W-MSR Resilient Consensus (F = 1)")
plt.xlabel("Time Step k")
plt.ylabel("State Value x_i[k]")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/wmsr_plot.png", dpi=150)
plt.show()
