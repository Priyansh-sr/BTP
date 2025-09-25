import gurobipy as gp
from gurobipy import GRB

# Jobs data: each job has a sequence of (machine, processing time)
jobs = {
    1: [(1, 3), (2, 2)],
    2: [(2, 2), (1, 1)],
    3: [(1, 2), (2, 1)],
}

machines = {1, 2}

# Total number of stages (simplified upper bound)
num_stages = 10

model = gp.Model("StagewiseJobShop")

# Binary decision: x[job, operation, stage] = 1 if operation starts at stage
x = {}
for j in jobs:
    for k in range(len(jobs[j])):
        for t in range(num_stages):
            x[j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{k}_{t}")

# Start time variables for each operation
S = {}
for j in jobs:
    for k in range(len(jobs[j])):
        S[j, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"S_{j}_{k}")

# Completion time variable
Cmax = model.addVar(lb=0, name="Cmax")

# Ensure each operation is assigned exactly one stage
for j in jobs:
    for k in range(len(jobs[j])):
        model.addConstr(gp.quicksum(x[j, k, t] for t in range(num_stages)) == 1, name=f"assign_{j}_{k}")

# Enforce sequential operations for each job across stages
for j in jobs:
    for k in range(len(jobs[j]) - 1):
        for t in range(num_stages):
            # operation k+1 can't start before k completes
            model.addConstr(
                gp.quicksum(t_prime * x[j, k + 1, t_prime] for t_prime in range(num_stages)) >=
                gp.quicksum((t + 1) * x[j, k, t] for t in range(num_stages)),
                name=f"seq_{j}_{k}"
            )

# Machine capacity: no two operations on the same machine overlap at same stage
for m in machines:
    for t in range(num_stages):
        model.addConstr(
            gp.quicksum(x[j, k, t] for j in jobs for k, (machine, _) in enumerate(jobs[j]) if machine == m) <= 1,
            name=f"mach_{m}_{t}"
        )

# Calculate start times based on assigned stages and processing times
bigM = 10000  # Large constant
for j in jobs:
    for k, (m, p) in enumerate(jobs[j]):
        model.addConstr(
            S[j, k] == gp.quicksum(t * p * x[j, k, t] for t in range(num_stages)),
            name=f"starttime_{j}_{k}"
        )

# Makespan is max completion time over all jobs and operations
for j in jobs:
    last_op_index = len(jobs[j]) - 1
    processing_time = jobs[j][last_op_index][1]
    model.addConstr(
        Cmax >= S[j, last_op_index] + processing_time,
        name=f"makespan_{j}"
    )

# Objective: minimize makespan
model.setObjective(Cmax, GRB.MINIMIZE)

model.optimize()

# Output results
for j in jobs:
    for k in range(len(jobs[j])):
        for t in range(num_stages):
            if x[j, k, t].X > 0.5:
                print(f"Job {j} Operation {k} starts at stage {t}, time {t * jobs[j][k][1]}")
print(f"Optimized makespan: {Cmax.X}")
