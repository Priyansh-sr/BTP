# btp_paraSDM_obj.py
# Para-SDM-style objective integrated into the node-based Job Shop MILP
# Uses linearization: diff = max(theta[k,nt] - theta[k,nf], 0), z = assign * diff
# Objective: sum(assign * (proc_time + diff)) = sum(proc_time * assign + z)

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import math

# ----------------------------
# Problem data
# ----------------------------
jobs = {
    1: [(1, 8),  (2, 5),  (3, 7),  (4, 6)],
    2: [(2, 6),  (3, 9),  (4, 5),  (1, 7)],
    3: [(3, 7),  (4, 4),  (1, 6),  (2, 8)],
    4: [(4, 5),  (1, 9),  (2, 7),  (3, 6)],
}



machines = {1, 2, 3,4}
K = sum(len(seq) for seq in jobs.values())  # number of stages

# ----------------------------
# Node generator
# ----------------------------
def make_nodes():
    nodes = {k: [] for k in range(K + 2)}
    for j in jobs:
        nodes[0].append((j, "start"))
    for k in range(1, K + 1):
        for j, seq in jobs.items():
            for op_idx, (m, _) in enumerate(seq, start=1):
                nodes[k].append((j, "M", op_idx))
                nodes[k].append((j, "W", op_idx))
            nodes[k].append((j, "C"))
        nodes[k].append(("C_global",))
    nodes[K + 1] = [("C_global",)]
    return nodes

nodes = make_nodes()

# ----------------------------
# Transition feasibility
# ----------------------------
def valid_transition(job, nf, nt, seq):
    if nf == (job, "start"):
        return nt in [(job, "M", 1), (job, "W", 1)]
    if len(nf) >= 3 and nf[1] == "M":
        op_idx = nf[2]
        if nt == nf:
            return False
        if nt == (job, "W", op_idx):
            return True
        if op_idx < len(seq):
            return nt in [(job, "M", op_idx + 1), (job, "W", op_idx + 1)]
        if op_idx == len(seq):
            return nt == (job, "C")
        return False
    if len(nf) >= 3 and nf[1] == "W":
        op_idx = nf[2]
        return nt in [(job, "W", op_idx), (job, "M", op_idx)]
    if nf == (job, "C"):
        return nt in [(job, "C"), ("C_global",)]
    if nf == ("C_global",):
        return nt == ("C_global",)
    return False

# ----------------------------
# Plot helper
# ----------------------------
def plot_machine_timeline(all_ops, title="Machine timeline (Gantt chart)"):
    if not all_ops:
        print("No operations to plot.")
        return
    jobs_list = sorted({op['job'] for op in all_ops})
    machines_list = sorted({op['machine'] for op in all_ops})
    mach_to_y = {m: i for i, m in enumerate(machines_list)}
    try:
        cmap = plt.colormaps.get_cmap("tab10")
    except Exception:
        cmap = plt.cm.get_cmap("tab10")
    job_colors = {j: cmap(i % 10) for i, j in enumerate(jobs_list)}
    height_per_machine = 0.6
    fig_height = max(3, len(machines_list) * (height_per_machine + 0.1))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    for op in all_ops:
        m = op['machine']
        y = mach_to_y[m]
        start = op['start'] if op['start'] is not None else 0.0
        finish = op['finish'] if op['finish'] is not None else (start + op['proc'])
        width = max(0.01, finish - start)
        color = job_colors[op['job']]
        ax.barh(y=y, width=width, left=start, height=0.6, align='center', color=color, edgecolor='black')
        label = f"J{op['job']}-O{op['op_idx']}"
        if width >= 0.8:
            ax.text(start + 0.1, y, label, va='center', ha='left', fontsize=8, color='white')
        else:
            ax.text(finish + 0.05, y, label, va='center', ha='left', fontsize=8, color='black')
    ax.set_yticks([mach_to_y[m] for m in machines_list])
    ax.set_yticklabels([f"Machine {m}" for m in machines_list])
    ax.set_xlabel("Time")
    ax.set_title(title)
    all_times = []
    for op in all_ops:
        if op['start'] is not None: all_times.append(op['start'])
        if op['finish'] is not None: all_times.append(op['finish'])
    if all_times:
        max_time = math.ceil(max(all_times))
        ax.set_xticks(range(0, max_time + 1, 1))
        ax.set_xlim(0, max_time + 0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Build model
# ----------------------------
model = gp.Model("paraSDM_node_theta_obj")
model.Params.OutputFlag = 1

# Big-M bounds
M = sum(p for seq in jobs.values() for (_, p) in seq) + 1000

# assign binary vars
assign = {}
for k in range(K + 1):
    for j, seq in jobs.items():
        for nf in nodes[k]:
            for nt in nodes[k + 1]:
                if valid_transition(j, nf, nt, seq):
                    assign[(k, nf, nt, j)] = model.addVar(vtype=GRB.BINARY, name=f"a_{k}_{nf}_{nt}_j{j}")

# theta vars
theta = {}
for k in nodes:
    for n in nodes[k]:
        theta[(k, n)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_{k}_{n}")

model.update()

# fix start times
for j in jobs:
    model.addConstr(theta[(0, (j, "start"))] == 0.0, name=f"start_time_zero_j{j}")

# time propagation constraints (big-M)
for key, a in list(assign.items()):
    k, nf, nt, j = key
    proc_time = 0.0
    if len(nt) >= 3 and nt[1] == "M":
        jobid = nt[0]; op_idx = nt[2]
        if jobid in jobs and 1 <= op_idx <= len(jobs[jobid]):
            proc_time = jobs[jobid][op_idx - 1][1]
    model.addConstr(theta[(k + 1, nt)] >= theta[(k, nf)] + proc_time - M * (1 - a),
                    name=f"timeprop_k{k}_{nf}_to_{nt}_j{j}")

# flow & selection constraints
for k in range(K + 1):
    for j in jobs:
        for nf in nodes[k]:
            outs = [assign[(k, nf, nt, j)] for nt in nodes[k + 1] if (k, nf, nt, j) in assign]
            if outs:
                model.addConstr(gp.quicksum(outs) <= 1, name=f"out_{k}_{j}_{nf}")

for j in jobs:
    start_out = [assign[(0, (j, "start"), nt, j)] for nt in nodes[1] if (0, (j, "start"), nt, j) in assign]
    model.addConstr(gp.quicksum(start_out) == 1, name=f"start_exact1_{j}")

for k in range(1, K + 1):
    for j in jobs:
        for n in nodes[k]:
            inflow = [assign[(k - 1, nf, n, j)] for nf in nodes[k - 1] if (k - 1, nf, n, j) in assign]
            outflow = [assign[(k, n, nt, j)] for nt in nodes[k + 1] if (k, n, nt, j) in assign]
            if inflow or outflow:
                model.addConstr(gp.quicksum(inflow) == gp.quicksum(outflow), name=f"flow_{k}_{j}_{n}")

for j in jobs:
    inflow_to_global = [assign[(k, nf, ("C_global",), j)] for k in range(K + 1) for nf in nodes[k] if (k, nf, ("C_global",), j) in assign]
    model.addConstr(gp.quicksum(inflow_to_global) == 1, name=f"job{j}_to_globalC")

# machine disjunctive pairwise constraints
machine_arcs = []
for key, a in assign.items():
    k, nf, nt, j = key
    if len(nt) >= 3 and nt[1] == "M":
        jobid = nt[0]; op_idx = nt[2]
        if jobid in jobs and 1 <= op_idx <= len(jobs[jobid]):
            machine_id = jobs[jobid][op_idx - 1][0]
            machine_arcs.append((key, machine_id, k, nf, nt, a))

order = {}
n_arcs = len(machine_arcs)
for i in range(n_arcs):
    key_i, m_i, k_i, nf_i, nt_i, a_i = machine_arcs[i]
    for j_idx in range(i + 1, n_arcs):
        key_j, m_j, k_j, nf_j, nt_j, a_j = machine_arcs[j_idx]
        if m_i != m_j:
            continue
        y = model.addVar(vtype=GRB.BINARY, name=f"ord_{i}_{j_idx}")
        order[(key_i, key_j)] = y
        model.addConstr(theta[(k_i + 1, nt_i)] <= theta[(k_j, nf_j)] + M * (1 - y) + M * (2 - a_i - a_j),
                        name=f"disj_i_before_j_{i}_{j_idx}")
        model.addConstr(theta[(k_j + 1, nt_j)] <= theta[(k_i, nf_i)] + M * (y) + M * (2 - a_i - a_j),
                        name=f"disj_j_before_i_{i}_{j_idx}")

# ----------------------------
# Para-SDM objective construction
# For each assign arc:
#   diff >= theta[k,nt] - theta[k,nf]    (diff >= 0 by lb)
#   z  = assign * diff  (linearized)
# Objective term: proc_time * assign + z
# ----------------------------
diff_vars = {}
z_vars = {}
obj = gp.LinExpr()

for key, a in assign.items():
    k, nf, nt, j = key
    # processing time at nt (if machine)
    proc_time = 0.0
    if len(nt) >= 3 and nt[1] == "M":
        jobid = nt[0]; op_idx = nt[2]
        proc_time = jobs[jobid][op_idx - 1][1]
    # diff variable (>= max(theta[k,nt] - theta[k,nf], 0))
    diff = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"diff_{k}_{nf}_{nt}_j{j}")
    diff_vars[key] = diff
    # diff constraint
    model.addConstr(diff >= theta[(k + 1, nt)] - theta[(k, nf)], name=f"diff_constr_{k}_{nf}_{nt}_j{j}")

    # z variable for assign * diff
    z = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"z_{k}_{nf}_{nt}_j{j}")
    z_vars[key] = z
    # linearization constraints for z = a * diff (a binary)
    # z <= M * a
    model.addConstr(z <= M * a, name=f"z_le_Ma_{k}_{nf}_{nt}_j{j}")
    # z <= diff
    model.addConstr(z <= diff, name=f"z_le_diff_{k}_{nf}_{nt}_j{j}")
    # z >= diff - M*(1 - a)
    model.addConstr(z >= diff - M * (1 - a), name=f"z_ge_diff_minus_M1ma_{k}_{nf}_{nt}_j{j}")
    # z >= 0 implicitly by lb
    # add term proc_time * a + z to objective
    if proc_time != 0:
        obj += proc_time * a
    obj += z

model.setObjective(obj, GRB.MINIMIZE)

# ----------------------------
# Solve
# ----------------------------
model.Params.TimeLimit = 120  # increase if you want longer search
model.Params.OutputFlag = 1
model.optimize()

# ----------------------------
# Extraction helpers
# ----------------------------
def reconstruct_path_for_job(assign_dict, j):
    path = []
    current_node = (j, "start")
    for k in range(0, K + 1):
        found = False
        for nt in nodes[k + 1]:
            key = (k, current_node, nt, j)
            if key in assign_dict and assign_dict[key].X > 0.5:
                path.append((k, current_node, nt))
                current_node = nt
                found = True
                break
        if not found:
            break
        if current_node == ("C_global",):
            break
    return path

def get_ops_for_job(assign_dict, theta_dict, j):
    path = reconstruct_path_for_job(assign_dict, j)
    ops = []
    for (k, nf, nt) in path:
        if len(nt) >= 3 and nt[1] == "M":
            jobid = nt[0]; op_idx = nt[2]
            machine_id = jobs[jobid][op_idx - 1][0]
            proc_time = jobs[jobid][op_idx - 1][1]
            t_start = theta_dict[(k, nf)].X if (k, nf) in theta_dict else None
            t_finish = theta_dict[(k + 1, nt)].X if (k + 1, nt) in theta_dict else None
            ops.append({
                'job': j, 'op_idx': op_idx, 'machine': machine_id,
                'k': k, 'nf': nf, 'nt': nt,
                'start': t_start, 'finish': t_finish, 'proc': proc_time
            })
    return path, ops

# ----------------------------
# Results and printing
# ----------------------------
status = model.Status
if status != GRB.OPTIMAL and status != GRB.TIME_LIMIT and status != GRB.SUBOPTIMAL:
    print("No feasible/optimal solution. Status:", status)
else:
    print("\nPara-SDM Objective value = %.6f\n" % model.ObjVal)

    # collect chosen arcs
    chosen_arcs = {}
    used_nodes = set()
    for key, var in assign.items():
        val = 0.0
        try:
            val = var.X
        except Exception:
            val = 0.0
        if val is not None and val > 0.5:
            k, nf, nt, j = key
            chosen_arcs.setdefault((k, j), []).append((nf, nt))
            used_nodes.add((k, nf)); used_nodes.add((k + 1, nt))

    # print per-job path and build ops
    all_ops = []
    print("--- Chosen paths per job ---")
    for j in sorted(jobs.keys()):
        path, ops = get_ops_for_job(assign, theta, j)
        print(f"\nJob {j} path (in order):")
        if not path:
            print("  (no path found for this job)")
            continue
        for (k, nf, nt) in path:
            t_nf = theta[(k, nf)].X if (k, nf) in theta else None
            t_nt = theta[(k + 1, nt)].X if (k + 1, nt) in theta else None
            t_nf_s = f"{t_nf:.2f}" if t_nf is not None else "N/A"
            t_nt_s = f"{t_nt:.2f}" if t_nt is not None else "N/A"
            print(f"  Stage {k}->{k+1}: {nf} -> {nt} | theta[{k}][{nf}]={t_nf_s} theta[{k+1}][{nt}]={t_nt_s}")
        all_ops.extend(ops)

    # stage-wise theta print
    print("\n--- Stage-wise theta for nodes on chosen paths ---")
    for k in sorted(nodes.keys()):
        stage_nodes = [n for n in nodes[k] if (k, n) in used_nodes]
        if not stage_nodes:
            continue
        print(f"\nStage {k}:")
        for n in stage_nodes:
            t = theta[(k, n)].X if (k, n) in theta else None
            t_s = f"{t:.2f}" if t is not None else "N/A"
            print(f"  theta[{k}][{n}] = {t_s}")

    # merged timeline and plot
    if all_ops:
        all_ops_sorted = sorted(all_ops, key=lambda x: (x['start'] if x['start'] is not None else 1e9, x['finish'] if x['finish'] is not None else 1e9))
        print("\n--- Merged timeline (operations sorted by start) ---")
        for o in all_ops_sorted:
            s = f"{o['start']:.2f}" if o['start'] is not None else "N/A"
            f = f"{o['finish']:.2f}" if o['finish'] is not None else "N/A"
            print(f"  {s} - {f}: Job{o['job']} Op{o['op_idx']} on M{o['machine']} (dur={o['proc']})")
        plot_machine_timeline(all_ops_sorted, title="Merged operation timeline")
    else:
        print("\nNo machine operations found on chosen paths.")

    # makespan (global completion)
    if (K + 1, ("C_global",)) in theta:
        try:
            makespan_val = theta[(K + 1, ("C_global",))].X
            print(f"\nGlobal completion theta[K+1][('C_global',)] = {makespan_val:.2f}")
        except Exception:
            print("\nGlobal completion theta not available.")
