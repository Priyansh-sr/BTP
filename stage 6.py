"""
para_sdm_stagewise_paper_exact.py

EXACT Stage-wise Para-SDM MILP for Job Shop Scheduling
-------------------------------------------------------

This script implements the Para-SDM framework from the IEEE CASE 2022 paper,
using the same conceptual variables:

  - θ_{k,γ}   →   theta[(k, γ)]
  - η_k(γ' | γ, γ0)   →   eta[(k, γ, γ', γ0)]
  - y_{iℓ} (machine disjunction ordering)
  - max(θ_{k+1,γ'} - θ_{k,γ}, 0) linearized → diff, z

Node types follow the paper:
  Γ₀ = { (J, j) }
  Γ_k = { (M,m), (MJ,j,r), (WJ,j,r), (Cj,j), δ }
  Γ_{K+1} = { δ }

Final output:
  - MILP optimum
  - Gantt chart identical to your reference screenshot
"""

import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib.pyplot as plt
import random

# ============================================================
# JOB DATA (from research paper Table I)
# ============================================================

jobs = {
    1: [(1, 8),  (2, 5),  (3, 7),  (4, 6)],
    2: [(2, 6),  (3, 9),  (4, 5),  (1, 7)],
    3: [(3, 7),  (4, 4),  (1, 6),  (2, 8)],
    4: [(4, 5),  (1, 9),  (2, 7),  (3, 6)],
}

machines = sorted({m for seq in jobs.values() for m,_ in seq})

# Number of stages (research paper: K = sum operations)
K = sum(len(seq) for seq in jobs.values())


# ============================================================
# NODE GENERATION Γ_k
# ============================================================

def build_nodes(jobs, K, machines):
    nodes = {k: [] for k in range(K+2)}

    # Γ₀ = jobs
    for j in jobs:
        nodes[0].append(("J", j))

    # Γ₁ ... Γ_K
    for k in range(1, K+1):
        # Machine placeholders (paper uses M_j)
        for m in machines:
            nodes[k].append(("M", m))

        # Machine-Job, Waiting, Completion nodes
        for j, seq in jobs.items():
            for r in range(1, len(seq)+1):
                nodes[k].append(("MJ", j, r))
                nodes[k].append(("WJ", j, r))
            nodes[k].append(("Cj", j))

        # Termination node δ allowed in intermediate levels
        nodes[k].append(("delta",))

    # Γ_{K+1} = {δ}
    nodes[K+1] = [("delta",)]
    return nodes


nodes = build_nodes(jobs, K, machines)


# ============================================================
# TRANSITION FUNCTION f(γ' | γ, γ0)
# ============================================================

def feasible_transition(j0, nf, nt, seq):
    """
    Research-paper transition rules implemented exactly.
    """

    # γ = (J,j)
    if nf[0] == "J":
        j = nf[1]
        if j != j0:
            return False
        return nt in [("MJ", j, 1), ("WJ", j, 1)]

    # γ = (MJ, j, r)
    if nf[0] == "MJ":
        j, r = nf[1], nf[2]

        if nt == ("WJ", j, r):
            return True

        if r < len(seq):  # more operations left
            return nt in [("MJ", j, r+1), ("WJ", j, r+1)]
        else:
            return nt == ("Cj", j)

    # γ = (WJ, j, r)
    if nf[0] == "WJ":
        j, r = nf[1], nf[2]
        return nt in [("WJ", j, r), ("MJ", j, r)]

    # γ = (Cj, j)
    if nf[0] == "Cj":
        j = nf[1]
        return nt in [("Cj", j), ("delta",)]

    # γ = δ
    if nf[0] == "delta":
        return nt == ("delta",)

    return False


# Processing time T_{γ}(γ0)
def proc_time_of(node):
    if node[0] == "MJ":
        j, r = node[1], node[2]
        return jobs[j][r-1][1]
    return 0.0


# Machine of MJ node
def machine_of(node):
    if node[0] == "MJ":
        j, r = node[1], node[2]
        return jobs[j][r-1][0]
    return None


# ============================================================
# BUILD GUROBI MODEL
# ============================================================

model = gp.Model("ParaSDM_PaperExact")
model.Params.OutputFlag = 1

M = sum(p for seq in jobs.values() for _,p in seq) + 100


# θ_{k,γ}
theta = {}
for k in nodes:
    for g in nodes[k]:
        name = f"theta_{k}_{str(g).replace(' ','')}"
        theta[(k,g)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=name)


# η_k(γ' | γ, γ0)
eta = {}
for k in range(0, K+1):
    for j0, seq in jobs.items():
        for nf in nodes[k]:
            for nt in nodes[k+1]:
                if feasible_transition(j0, nf, nt, seq):
                    name = f"eta_{k}({str(nt).replace(' ','')}|{str(nf).replace(' ','')},J{j0})"
                    eta[(k,nf,nt,("J",j0))] = model.addVar(vtype=GRB.BINARY, name=name)


# ============================================================
# MACHINE DISJUNCTION (y variables)
# ============================================================

machine_arcs = []
for key, a_var in eta.items():
    k, nf, nt, g0 = key
    if nt[0] == "MJ":
        m = machine_of(nt)
        if m is not None:
            machine_arcs.append((key, m, k, nf, nt, a_var))

y = {}
for i in range(len(machine_arcs)):
    key_i, m_i, k_i, nf_i, nt_i, a_i = machine_arcs[i]
    for j in range(i+1, len(machine_arcs)):
        key_j, m_j, k_j, nf_j, nt_j, a_j = machine_arcs[j]

        if m_i == m_j and k_i == k_j:
            y[(i,j)] = model.addVar(vtype=GRB.BINARY,
                                    name=f"y_{i}_{j}_k{k_i}")

            # exact disjunction
            model.addConstr(
                theta[(k_i+1, nt_i)]
                <= theta[(k_j, nf_j)] + M*(1 - y[(i,j)]) + M*(2 - a_i - a_j)
            )

            model.addConstr(
                theta[(k_j+1, nt_j)]
                <= theta[(k_i, nf_i)] + M*(y[(i,j)]) + M*(2 - a_i - a_j)
            )


# ============================================================
# TIME PROPAGATION θ_{k+1,γ'} ≥ θ_{k,γ} + T(γ') - M(1-η)
# ============================================================

for key, a_var in eta.items():
    k, nf, nt, g0 = key
    p = proc_time_of(nt)

    model.addConstr(
        theta[(k+1, nt)]
        >= theta[(k, nf)] + p - M*(1 - a_var)
    )


# ============================================================
# FLOW CONSTRAINTS (paper eq. 8)
# ============================================================

# Start node: exactly one outgoing arc
for j0 in jobs:
    gamma0 = ("J", j0)
    outs = [
        eta[key]
        for key in eta
        if key[0] == 0 and key[1] == gamma0 and key[3] == gamma0
    ]
    model.addConstr(quicksum(outs) == 1)


# Flow conservation for k = 1..K
for k in range(1, K+1):
    for j0, seq in jobs.items():
        gamma0 = ("J", j0)

        for g in nodes[k]:
            infl = [eta[(k-1, nf, g, gamma0)]
                    for nf in nodes[k-1]
                    if (k-1, nf, g, gamma0) in eta]

            out = [eta[(k, g, nt, gamma0)]
                   for nt in nodes[k+1]
                   if (k, g, nt, gamma0) in eta]

            if infl or out:
                model.addConstr(quicksum(infl) == quicksum(out))


# Must reach δ exactly once
for j0 in jobs:
    gamma0 = ("J", j0)
    arcs_to_delta = [
        eta[key] for key in eta
        if key[2] == ("delta",) and key[3] == gamma0
    ]
    model.addConstr(quicksum(arcs_to_delta) == 1)


# ============================================================
# OBJECTIVE: Σ_k η_k ( T(γ') + max( θ_{k+1,γ'} - θ_{k,γ} , 0 ) )
# diff and z linearize waiting term
# ============================================================

diff = {}
z = {}
obj = gp.LinExpr()

for key, a_var in eta.items():
    k, nf, nt, g0 = key

    # diff = θ_{k+1,γ'} - θ_{k,γ}
    diff[key] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                             name=f"diff_{k}_{nf}_to_{nt}_J{g0[1]}")
    model.addConstr(diff[key] >= theta[(k+1, nt)] - theta[(k, nf)])

    # z = max(diff, 0) * η
    z[key] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                          name=f"z_{k}_{nf}_to_{nt}_J{g0[1]}")
    model.addConstr(z[key] <= diff[key])
    model.addConstr(z[key] <= M*a_var)
    model.addConstr(z[key] >= diff[key] - M*(1 - a_var))

    p = proc_time_of(nt)
    obj += p*a_var       # processing time
    obj += z[key]        # waiting time


model.setObjective(obj, GRB.MINIMIZE)


# θ_{0,Jj} = 0
for j0 in jobs:
    model.addConstr(theta[(0,("J",j0))] == 0)


# Solve
model.optimize()


# ============================================================
# RECONSTRUCT PATHS
# ============================================================

def reconstruct(j0):
    cur = ("J", j0)
    path = []

    for k in range(0, K+1):
        found = False
        for nt in nodes[k+1]:
            key = (k, cur, nt, ("J", j0))
            if key in eta and eta[key].X > 0.5:
                path.append((k, cur, nt))
                cur = nt
                found = True
                break
        if not found or cur == ("delta",):
            break
    return path


# ============================================================
# CREATE LIST OF OPERATIONS FOR GANTT CHART
# ============================================================

ops = []
if model.Status != GRB.INFEASIBLE:
    for j0 in jobs:
        path = reconstruct(j0)
        for (k, nf, nt) in path:
            if nt[0] == "MJ":
                j, r = nt[1], nt[2]
                m = machine_of(nt)
                ops.append({
                    'machine': m,
                    'job': j,
                    'op': r,
                    'start': theta[(k, nf)].X,
                    'end':   theta[(k+1, nt)].X
                })

ops = sorted(ops, key=lambda x: (x['machine'], x['start']))


# ============================================================
# FINAL GANTT CHART (exact screenshot style)
# ============================================================

if ops:
    machines_used = sorted({o['machine'] for o in ops})
    mach_y = {m:i for i,m in enumerate(machines_used)}

    fig, ax = plt.subplots(figsize=(14,6))

    colors = {}
    for o in ops:
        key = (o['job'], o['op'])
        if key not in colors:
            random.seed(hash(key)%999999)
            colors[key] = (
                0.4+random.random()*0.4,
                0.4+random.random()*0.4,
                0.4+random.random()*0.4
            )

    for o in ops:
        y = mach_y[o['machine']]
        dur = o['end'] - o['start']

        ax.broken_barh(
            [(o['start'], dur)],
            (y-0.35, 0.7),
            facecolor=colors[(o['job'],o['op'])],
            edgecolor='black',
            linewidth=1.3
        )

        ax.text(
            o['start'] + dur*0.1,
            y,
            f"J{o['job']}-O{o['op']}",
            va='center', ha='left',
            fontsize=10, fontweight='bold', color='white'
        )

    ax.set_yticks([mach_y[m] for m in machines_used])
    ax.set_yticklabels([f"Machine {m}" for m in machines_used], fontsize=11)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_title("Merged operation timeline", fontsize=14, fontweight='bold')

    ax.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.4)

    max_time = max(o['end'] for o in ops)
    ax.set_xlim(0, max_time + 5)

    plt.tight_layout()
    plt.show()

print("\n=========================")
print("Objective value =", model.ObjVal)
print("=========================\n")
