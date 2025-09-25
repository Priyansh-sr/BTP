import gurobipy as gp
from gurobipy import GRB

# ----------------------------
# Problem Data
# ----------------------------
# Jobs dictionary: job_id : [(machine, processing_time), ...]
jobs = {
    1: [(1, 3), (2, 2)],   # Job 1: M1-3, then M2-2
    2: [(2, 2), (1, 1)]    # Job 2: M2-2, then M1-1
}

machines = [1, 2]          # List of machines
bigM = 1000                # Big-M value (upper bound on time)

# ----------------------------
# Model
# ----------------------------
model = gp.Model("JobShop")

# Decision variables
S = {}     # Start times
C = {}     # Completion times

for i in jobs:
    for k, (m, p) in enumerate(jobs[i]):
        S[i, k] = model.addVar(lb=0, name=f"S_{i}_{k}")
        C[i, k] = model.addVar(lb=0, name=f"C_{i}_{k}")

# Makespan
Cmax = model.addVar(lb=0, name="Cmax")

# ----------------------------
# Constraints
# ----------------------------
# 1. Completion time definition
for i in jobs:
    for k, (m, p) in enumerate(jobs[i]):
        model.addConstr(C[i, k] == S[i, k] + p)

# 2. Job operation order
for i in jobs:
    for k in range(len(jobs[i]) - 1):
        model.addConstr(S[i, k+1] >= C[i, k])

# 3. Machine non-overlap (disjunctive constraints with binaries)
for m in machines:
    # Get all operations on this machine
    ops = [(i, k) for i in jobs for k, (mm, _) in enumerate(jobs[i]) if mm == m]
    # Pairwise ordering
    for a in range(len(ops)):
        for b in range(a+1, len(ops)):
            i, k = ops[a]
            j, l = ops[b]
            x = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}_{j}_{l}")
            # Either (i,k) before (j,l) OR (j,l) before (i,k)
            model.addConstr(S[i, k] >= C[j, l] - bigM * (1 - x))
            model.addConstr(S[j, l] >= C[i, k] - bigM * x)

# 4. Makespan definition
for i in jobs:
    last_op = len(jobs[i]) - 1
    model.addConstr(Cmax >= C[i, last_op])

# ----------------------------
# Objective
# ----------------------------
model.setObjective(Cmax, GRB.MINIMIZE)

# ----------------------------
# Solve
# ----------------------------
model.optimize()

# ----------------------------
# Print Results
# ----------------------------
print("\nOptimal Schedule:")
for i in jobs:
    for k in range(len(jobs[i])):
        print(f"Job {i} Operation {k}: Start = {S[i, k].X}, End = {C[i, k].X}")

print(f"\nOptimal Makespan = {Cmax.X}")
