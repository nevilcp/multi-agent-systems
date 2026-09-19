"""
Problem 1 - formation control that spells out "NEVIL" one letter at a time.

20 agents talk to each other over a random (Erdos-Renyi) graph and drive
toward letter-shaped target positions using a consensus + anchoring law:
    pdot_i = - sum_{j in N_i} a_ij [ (p_i - p_i*) - (p_j - p_j*) ]  - kp (p_i - p_i*)
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import animation

np.random.seed(7)  # keep results repeatable across runs

N = 20
p_prob = 0.3
dt = 0.03
steps_per_letter = 260
kp = 0.6

# Erdos-Renyi graphs aren't always connected, so just keep rerolling until we get one.
while True:
    G = nx.erdos_renyi_graph(N, p_prob, seed=np.random.randint(0, 10_000))
    if nx.is_connected(G):
        break
L = nx.laplacian_matrix(G).toarray().astype(float)

# Letters are built from straight-line strokes (pen up between strokes, like
# tracing them by hand). We space points evenly along each stroke, with the
# number of points per stroke roughly proportional to its length, so every
# letter ends up with exactly N points to match the N agents.
LETTER_STROKES = {
    "N": [[(0, 0), (0, 6)], [(0, 6), (4, 0)], [(4, 0), (4, 6)]],
    "E": [[(0, 0), (0, 6)], [(0, 6), (3, 6)], [(0, 3), (2.2, 3)], [(0, 0), (3, 0)]],
    "V": [[(0, 6), (2, 0)], [(2, 0), (4, 6)]],
    "I": [[(1.3, 0), (2.7, 0)], [(2, 0), (2, 6)], [(1.3, 6), (2.7, 6)]],
    "L": [[(0, 6), (0, 0)], [(0, 0), (3, 0)]],
}


def generate_letter(letter, n_points, x_offset=0.0):
    strokes = LETTER_STROKES[letter]
    lengths = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in strokes]
    total_len = sum(lengths)
    # give each stroke points proportional to its length (min 2 so it's still a line)
    raw_alloc = [max(2, round(n_points * l / total_len)) for l in lengths]
    # rounding can push us a point or two off target, so nudge the biggest stroke
    while sum(raw_alloc) > n_points:
        raw_alloc[int(np.argmax(raw_alloc))] -= 1
    while sum(raw_alloc) < n_points:
        raw_alloc[int(np.argmax(raw_alloc))] += 1

    pts = []
    for (a, b), k in zip(strokes, raw_alloc):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        t = np.linspace(0, 1, k)
        seg = a[None, :] + t[:, None] * (b - a)[None, :]
        pts.append(seg)
    pts = np.vstack(pts)[:n_points]
    pts[:, 0] += x_offset
    return pts


NAME = "NEVIL"
targets = [generate_letter(ch, N) for ch in NAME]

# Run the actual simulation: agents start scattered and get pulled toward
# each letter in sequence, N -> E -> V -> I -> L.
p = np.random.uniform(-1, 7, size=(N, 2))  # scattered starting positions
trajectory = [p.copy()]
letter_boundaries = [0]  # index into trajectory where each letter's phase ends
err_history = []

for target in targets:
    for step in range(steps_per_letter):
        e = p - target
        dp = -(L @ e) - kp * e
        p = p + dp * dt
        trajectory.append(p.copy())
        err_history.append(np.linalg.norm(e) / N)
    letter_boundaries.append(len(trajectory) - 1)

trajectory = np.array(trajectory)
print("Final per-letter mean formation error (should be ~0):")
for i, ch in enumerate(NAME):
    idx = letter_boundaries[i + 1]
    e_final = np.linalg.norm(trajectory[idx] - targets[i]) / N
    print(f"  {ch}: {e_final:.5f}")

# Figure 1 - just the communication graph itself, so we can see the topology
fig_g, ax_g = plt.subplots(figsize=(4.2, 4.2))
pos = nx.spring_layout(G, seed=3)
nx.draw(
    G,
    pos,
    ax=ax_g,
    with_labels=True,
    node_color="#a8d0ff",
    edge_color="#555555",
    node_size=380,
    font_size=8,
    font_weight="bold",
)
ax_g.set_title(f"Erdos-Renyi communication graph $\\mathcal{{G}}$ (N={N}, p={p_prob})")
fig_g.tight_layout()
fig_g.savefig("figs/p1_graph.png", dpi=200)
plt.close(fig_g)

# Figure 2 - snapshot of where the agents end up for each letter, with the
# intended stroke outline shown faintly underneath for comparison
fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
for i, ch in enumerate(NAME):
    idx = letter_boundaries[i + 1]
    axes[i].scatter(
        trajectory[idx][:, 0],
        trajectory[idx][:, 1],
        c="#1f5fb8",
        s=45,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    for a, b in LETTER_STROKES[ch]:
        axes[i].plot([a[0], b[0]], [a[1], b[1]], color="#cccccc", lw=8, zorder=1)
    axes[i].set_xlim(-1, 5)
    axes[i].set_ylim(-1, 7)
    axes[i].set_aspect("equal")
    axes[i].set_title(f"'{ch}'", fontsize=13)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
fig.suptitle("Converged agent formations spelling NEVIL", y=1.03)
fig.tight_layout()
fig.savefig("figs/p1_letters.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure 3 - error over time, with dashed lines marking where each letter starts
fig_e, ax_e = plt.subplots(figsize=(7.5, 3.2))
ax_e.plot(err_history, color="#1f5fb8", lw=1.3)
for i, b in enumerate(letter_boundaries[1:-1]):
    ax_e.axvline(b, color="gray", ls="--", lw=0.8)
tick_pos = [
    (letter_boundaries[i] + letter_boundaries[i + 1]) / 2 for i in range(len(NAME))
]
ax_e.set_xticks(tick_pos)
ax_e.set_xticklabels(list(NAME))
ax_e.set_ylabel(r"$\frac{1}{N}\|p-p^*\|$")
ax_e.set_xlabel("Formation phase")
ax_e.set_title(
    "Mean formation error through the N\u2192E\u2192V\u2192I\u2192L sequence"
)
ax_e.grid(alpha=0.3)
fig_e.tight_layout()
fig_e.savefig("figs/p1_error.png", dpi=200)
plt.close(fig_e)

# Animation - saved as mp4 so it's easy to drop into the GitHub submission
fig_a, ax_a = plt.subplots(figsize=(5, 5))
scat = ax_a.scatter(trajectory[0][:, 0], trajectory[0][:, 1], c="#1f5fb8", s=45)
ax_a.set_xlim(-2, 6)
ax_a.set_ylim(-2, 8)
ax_a.set_aspect("equal")
title = ax_a.set_title("NEVIL formation control")


def update(frame):
    scat.set_offsets(trajectory[frame])
    letter_idx = min(
        np.searchsorted(letter_boundaries, frame, side="right") - 1, len(NAME) - 1
    )
    title.set_text(f"Forming: {NAME[max(letter_idx,0)]}")
    return scat, title


ani = animation.FuncAnimation(
    fig_a, update, frames=range(0, len(trajectory), 3), interval=15, blit=False
)
ani.save("figs/p1_nevil_formation.mp4", writer="ffmpeg", fps=40, dpi=150)
plt.close(fig_a)

print("Problem 1 artifacts written to figs/")
