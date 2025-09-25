import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# Input: jobs, machines, ordered routes
# -----------------------------
routes = {
   1: [(5,0.65), (3,0.514), (7,0.64), (2,0.202), (1,0.202), (4,0.24), (6,0.29)],
    2: [(3,0.722), (2,0.338), (1,0.242), (8,0.242), (7,0.2), (4,0.2), (6,0.338), (5,0.578)],
    3: [(2,0.089), (6,0.117), (1,0.185), (3,0.485), (5,0.185), (8,0.369), (4,0.117), (7,0.089)],
    4: [(6,0.061), (2,0.061), (5,0.265), (3,0.025), (1,0.013), (4,0.025)],
    5: [(5,0.392), (6,0.128), (2,0.05), (3,0.128), (1,0.072), (4,0.072)],
    6: [(2,0.02), (5,0.244), (4,0.052), (3,0.052), (1,0.052), (6,0.01)],
    7: [(5,0.269), (4,0.185), (6,0.06), (2,0.065), (1,0.029), (7,0.029), (3,0.017)],
    8: [(3,0.074), (5,0.074), (2,0.034), (1,0.034), (4,0.29), (6,0.02)]
}
jobs = list(routes.keys())
machines = sorted({m for ops in routes.values() for (m, _) in ops})
K = sum(len(ops) for ops in routes.values())

# -----------------------------
# Model
# -----------------------------
model = gp.Model("para_mdp")

# Theta[k,m]: machine availability
Theta = model.addVars(range(K+1), machines, lb=0, name="Theta")

# eta[k,i,t]: binary, 1 if job i's operation t scheduled at stage k
ops = [(i,t) for i in jobs for t in range(len(routes[i]))]
eta = model.addVars(range(K), ops, vtype=GRB.BINARY, name="eta")

# Makespan
Cmax = model.addVar(lb=0, name="Cmax")

# -----------------------------
# Constraints
# -----------------------------

# (1) One operation scheduled per stage
for k in range(K):
    model.addConstr(gp.quicksum(eta[k,i,t] for (i,t) in ops) == 1)

# (2) Each operation executed exactly once
for (i,t) in ops:
    model.addConstr(gp.quicksum(eta[k,i,t] for k in range(K)) == 1)

# (3) Theta recursion: machine availability
for k in range(K):
    for m in machines:
        model.addConstr(
            Theta[k+1,m] >= Theta[k,m] +
            gp.quicksum(routes[i][t][1]*eta[k,i,t] for (i,t) in ops if routes[i][t][0]==m)
        )

# (4) Makespan
for (i,t) in ops:
    m = routes[i][t][0]
    model.addConstr(Cmax >= Theta[K,m])

# (5) Precedence constraints: operation t cannot start before t-1 is done
for i in jobs:
    for t in range(1, len(routes[i])):  # start from second operation
        for k2 in range(K):
            model.addConstr(
                gp.quicksum(eta[k1, i, t-1] for k1 in range(k2+1)) 
                >= eta[k2, i, t]
            )

# -----------------------------
# Objective
# -----------------------------
model.setObjective(Cmax, GRB.MINIMIZE)

# -----------------------------
# Solve
# -----------------------------
model.optimize()

# -----------------------------
# Display results
# -----------------------------
if model.status == GRB.OPTIMAL:
    print("\nOptimal solution found:")
    print(f"Makespan (Cmax) = {model.objVal}")
    print("\nSchedule:")

    # Iterate through each stage and identify the scheduled operation
    for k in range(K):
        for (i, t) in ops:
            if eta[k, i, t].X > 0.5:
                job_id = i
                op_index = t
                machine_id = routes[job_id][op_index][0]
                processing_time = routes[job_id][op_index][1]
                print(f"Stage {k+1}: Job {job_id}'s operation {op_index+1} on Machine {machine_id} for {processing_time} units")

    print("\nMachine busy times (Theta):")
    for k in range(K + 1):
        for m in machines:
            if Theta[k, m].X > 0:
                print(f"Theta[{k}, {m}] = {Theta[k, m].X}")
else:
    print("\nNo optimal solution found.")
    print(f"Status code: {model.status}")
