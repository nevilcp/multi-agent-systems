"""
Problem 1 - Erdos-Renyi random graphs.

Part 1: a small N = 5 graph - adjacency, degree, incidence and Laplacian
        matrices, Laplacian eigenvalues and the algebraic connectivity lambda_2.
Part 2: a connected N = 10^3 graph - highest-degree node, mean degree / mean
        edges against theory, the diameter path, and a random Dijkstra path.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

Path("figures").mkdir(exist_ok=True)
random.seed(42)  # only the random node pair in part 2 depends on this

# ==========================================
# Part 1: N = 5 Erdos-Renyi graph
# ==========================================
N_small = 5
p_small = 0.5
G_small = nx.erdos_renyi_graph(N_small, p_small, seed=42)

# (a) matrices
A = nx.adjacency_matrix(G_small).toarray()
D = np.diag([d for _, d in G_small.degree()])
Inc = nx.incidence_matrix(G_small).toarray()
L = nx.laplacian_matrix(G_small).toarray()

print("--- N=5 graph matrices ---")
print("Adjacency matrix:\n", A)
print("Degree matrix:\n", D)
print("Incidence matrix:\n", Inc)
print("Laplacian matrix:\n", L)

# (b) eigenvalues and algebraic connectivity
eigenvalues = np.sort(np.linalg.eigvalsh(L))
lambda_2 = eigenvalues[1]
print(f"Laplacian eigenvalues: {eigenvalues}")
print(f"Algebraic connectivity (lambda_2): {lambda_2:.4f}\n")

plt.figure(figsize=(4, 4))
nx.draw(G_small, with_labels=True, node_color="lightblue", font_weight="bold")
plt.title("N=5 Erdos-Renyi graph")
plt.savefig("figures/p1_er_n5.png", dpi=200, bbox_inches="tight")
plt.show()

# ==========================================
# Part 2: N = 1000 Erdos-Renyi graph
# ==========================================
N_large = 1000
p_large = 0.01

# an Erdos-Renyi graph isn't always connected, so reroll (deterministically) until it is
graph_seed = 42
while True:
    G_large = nx.erdos_renyi_graph(N_large, p_large, seed=graph_seed)
    if nx.is_connected(G_large):
        break
    graph_seed += 1

# (a) highest-degree node and mean degree / mean edges against theory
degrees = dict(G_large.degree())
max_degree_node = max(degrees, key=degrees.get)

empirical_mean_deg = sum(degrees.values()) / N_large
empirical_mean_edges = G_large.number_of_edges()
theoretical_mean_deg = (N_large - 1) * p_large
theoretical_mean_edges = p_large * (N_large * (N_large - 1)) / 2

print("--- N=1000 graph properties ---")
print(f"Empirical mean degree: {empirical_mean_deg} | Theoretical: {theoretical_mean_deg}")
print(f"Empirical mean edges:  {empirical_mean_edges} | Theoretical: {theoretical_mean_edges}")

# diameter path: farthest node from a peripheral node
source_node = nx.periphery(G_large)[0]
path_lengths = nx.single_source_shortest_path_length(G_large, source_node)
target_node = max(path_lengths, key=path_lengths.get)
diameter_path = nx.shortest_path(G_large, source_node, target_node)
diameter_edges = list(zip(diameter_path, diameter_path[1:], strict=False))
print(f"Diameter: {nx.diameter(G_large)} (path of {len(diameter_path) - 1} edges)")

# (b) Dijkstra shortest path between two random nodes
rand_start, rand_target = random.sample(list(G_large.nodes()), 2)
dijkstra_path = nx.dijkstra_path(G_large, rand_start, rand_target)
dijkstra_edges = list(zip(dijkstra_path, dijkstra_path[1:], strict=False))
print(f"Dijkstra path {rand_start} -> {rand_target}: {len(dijkstra_path) - 1} edges")

pos = nx.spring_layout(G_large, seed=42)
plt.figure(figsize=(10, 10))

# everything very lightly, then the highlights on top
nx.draw_networkx_nodes(G_large, pos, node_size=10, node_color="gray", alpha=0.3)
nx.draw_networkx_edges(G_large, pos, alpha=0.02)

nx.draw_networkx_nodes(G_large, pos, nodelist=[max_degree_node], node_size=60, node_color="red")

nx.draw_networkx_nodes(G_large, pos, nodelist=diameter_path, node_size=30, node_color="orange")
nx.draw_networkx_edges(G_large, pos, edgelist=diameter_edges, edge_color="orange", width=3)

nx.draw_networkx_nodes(G_large, pos, nodelist=dijkstra_path, node_size=30, node_color="green")
nx.draw_networkx_edges(G_large, pos, edgelist=dijkstra_edges, edge_color="green", width=3)

plt.title("N=1000 ER graph\nRed: max degree | Orange: diameter | Green: random shortest path")
plt.axis("off")
plt.savefig("figures/p1_er_n1000.png", dpi=200, bbox_inches="tight")
plt.show()
