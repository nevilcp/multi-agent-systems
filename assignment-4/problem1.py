import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

Path("figures").mkdir(exist_ok=True)

with open("data/epidemic_data.csv") as f:
    rows = list(csv.DictReader(f))
N = 300
K = max(int(r["day"]) for r in rows) + 1
st = np.zeros((K, N), dtype="U1")  # st[day, node] = 'S'/'I'/'R'
for r in rows:
    st[int(r["day"]), int(r["node"]) - 1] = r["state"]

A = np.zeros((N, N))  # symmetric adjacency (undirected)
with open("data/network.csv") as f:
    edges = list(csv.DictReader(f))
for r in edges:
    i, j = int(r["source"]) - 1, int(r["target"]) - 1
    A[i, j] = A[j, i] = 1.0
print("days", K, "edges", int(A.sum() / 2))

S = (st == "S").sum(1)
I = (st == "I").sum(1)
R = (st == "R").sum(1)
k = np.arange(K)
plt.plot(k, S, label="S")
plt.plot(k, I, label="I")
plt.plot(k, R, label="R")
plt.xlabel("day")
plt.ylabel("individuals")
plt.legend()
plt.title("Observed SIR counts")
plt.savefig("figures/p1_sir_counts.png", dpi=150)
plt.show()
print("(a) max infected      :", I.max())
print("(b) day of max        :", I.argmax())
print("(c) ever infected     :", N - S[-1])
print("(d) fraction infected :", (N - S[-1]) / N)

Kg = [t for t in range(K - 1) if I[t] > 0]
gamma = np.mean([(R[t + 1] - R[t]) / I[t] for t in Kg])
print("gamma =", gamma, "| mean infectious duration =", 1 / gamma, "days")

Kb = [t for t in range(K - 1) if S[t] * I[t] > 0]
beta = np.mean([(S[t] - S[t + 1]) / (S[t] * I[t]) for t in Kb])
R0 = beta * N / gamma
print("beta =", beta, "| R0 =", R0)

def sir(t, y):
    s, i, r = y
    return [-beta * s * i, beta * s * i - gamma * i, gamma * i]


sol = solve_ivp(sir, [0, K - 1], [S[0], I[0], R[0]], t_eval=k, rtol=1e-8)
for y, c, n in zip(sol.y, "bgr", "SIR", strict=True):
    plt.plot(k, y, c + "-", label=n + " model")
for y, c, n in zip((S, I, R), "bgr", "SIR", strict=True):
    plt.plot(k, y, c + ".", ms=3, label=n + " data")
plt.xlabel("day")
plt.legend()
plt.title("Fitted SIR vs data")
plt.savefig("figures/p1_sir_fit.png", dpi=150)
plt.show()
print("RMSE (S, I, R):", np.sqrt(np.mean((sol.y - np.array([S, I, R])) ** 2, 1)))
print("model peak:", sol.y[1].max(), "on day", sol.y[1].argmax(), "| model final S:", sol.y[0][-1])

def simulate(p, vacc, rng):
    s = (st[0] == "S").copy()
    i = (st[0] == "I").copy()
    v = np.zeros(N, bool)
    v[list(vacc)] = True
    s &= ~v
    i &= ~v
    Ik = [i.sum()]
    for _ in range(K - 1):
        n_inf = A @ i.astype(float)
        new = s & (rng.random(N) < 1 - (1 - p) ** n_inf)
        rec = i & (rng.random(N) < gamma)
        s = s & ~new
        i = (i & ~rec) | new
        Ik.append(i.sum())
    return np.array(Ik), s.sum()


rng = np.random.default_rng(0)
target = N - S[-1]
ps = np.linspace(0.005, 0.2, 40)
fs = [np.mean([N - simulate(p, [], rng)[1] for _ in range(30)]) for p in ps]
p = ps[np.argmin(np.abs(np.array(fs) - target))]
print("calibrated p =", p)

B = 15
deg = A.sum(1)
evals, evecs = np.linalg.eigh(A)
ev = np.abs(evecs[:, -1])  # dominant eigenvector
top_deg = np.argsort(-deg)[:B]
top_eig = np.argsort(-ev)[:B]
print(
    "lambda_max =",
    evals[-1],
    "| mean degree =",
    deg.mean(),
    "| overlap of top-15 sets:",
    len(set(top_deg) & set(top_eig)),
)

strategies = {"None": [], "Degree": top_deg, "Eigenvector": top_eig, "Random": None}
curves = {}
for name, vac in strategies.items():
    runs, nvac = [], (B if vac is None else len(vac))
    for _ in range(300):
        v = rng.choice(N, B, replace=False) if vac is None else vac
        runs.append(simulate(p, v, rng))
    Ia = np.array([r[0] for r in runs])
    n_inf = np.array([N - r[1] - nvac for r in runs])  # excludes vaccinated nodes
    curves[name] = Ia.mean(0)
    print(
        f"{name:12s} Imax={Ia.max(1).mean():6.1f}  "
        f"Tpeak={Ia.argmax(1).mean():5.1f}  Ninfected={n_inf.mean():6.1f}"
    )

for n, c in curves.items():
    plt.plot(c, label=n)
plt.xlabel("day")
plt.ylabel("mean I")
plt.legend()
plt.title("Vaccination strategies (mean of 300 runs)")
plt.savefig("figures/p1_vaccination.png", dpi=150)
plt.show()
