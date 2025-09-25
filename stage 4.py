import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# Input: jobs, machines, ordered routes
# -----------------------------
routes = {
    1: [(1, 3), (2, 2)],   # Job 1: M1(3) -> M2(2)
    2: [(2, 2), (1, 4)]    # Job 2: M2(2) -> M1(4)
}
jobs = list(routes.keys())
machines = sorted({m for ops in routes.values() for (m, _) in ops})
K = sum(len(ops) for ops in routes.values())  # total operations
S = K  # number of stage slots allowed

# safe Big-M
total_proc = sum(p for ops in routes.values() for (_, p) in ops)
M = total_proc

# -----------------------------
# Model
# -----------------------------
model = gp.Model("para_mdp_multiperstage_precedence_with_wait")

# Theta[k,m] for k in 0..S
Theta = model.addVars(range(S+1), machines, lb=0.0, name="Theta")

# eta[k,i,t] = 1 if job i, operation t is placed in stage k
eta = model.addVars(
    [(k,i,t) for k in range(S) for i in jobs for t in range(len(routes[i]))],
    vtype=GRB.BINARY, name="eta"
)

# makespan
Cmax = model.addVar(lb=0.0, name="Cmax")

# waiting variables W[i,t] between op t and t+1
W = {}
for i in jobs:
    for t in range(len(routes[i]) - 1):
        W[i, t] = model.addVar(lb=0.0, name=f"W_{i}_{t}")

# job completion variables
Cjob = {i: model.addVar(lb=0.0, name=f"Cjob_{i}") for i in jobs}

model.update()

# -----------------------------
# Constraints
# -----------------------------

# Theta[0,m] = 0
for m in machines:
    model.addConstr(Theta[0, m] == 0.0)

# Machine capacity per stage
for k in range(S):
    for m in machines:
        model.addConstr(
            gp.quicksum(eta[k, i, t]
                        for i in jobs for t in range(len(routes[i]))
                        if routes[i][t][0] == m) <= 1
        )

# Each op executed exactly once
for i in jobs:
    for t in range(len(routes[i])):
        model.addConstr(gp.quicksum(eta[k, i, t] for k in range(S)) == 1)

# Theta recursion
for k in range(S):
    for m in machines:
        model.addConstr(
            Theta[k + 1, m] >= Theta[k, m] +
            gp.quicksum(routes[i][t][1] * eta[k, i, t]
                        for i in jobs for t in range(len(routes[i]))
                        if routes[i][t][0] == m)
        )

# Makespan lower bound
for m in machines:
    model.addConstr(Cmax >= Theta[S, m])

# -----------------------------
# Exact precedence with Big-M
# -----------------------------
for i in jobs:
    for t in range(len(routes[i]) - 1):
        m1 = routes[i][t][0]
        m2 = routes[i][t+1][0]
        for k1 in range(S):
            for k2 in range(S):
                model.addConstr(
                    Theta[k2, m2] + M * (2 - eta[k1, i, t] - eta[k2, i, t+1])
                    >= Theta[k1 + 1, m1]
                )

# -----------------------------
# Waiting linearization
# -----------------------------
for i in jobs:
    for t in range(len(routes[i]) - 1):
        m1 = routes[i][t][0]
        m2 = routes[i][t+1][0]
        for k1 in range(S):
            for k2 in range(S):
                model.addConstr(
                    W[i, t] + M * (2 - eta[k1, i, t] - eta[k2, i, t+1])
                    >= Theta[k2, m2] - Theta[k1 + 1, m1]
                )

# -----------------------------
# Job completion
# -----------------------------
for i in jobs:
    last = len(routes[i]) - 1
    m_last = routes[i][last][0]
    for k in range(S):
        model.addConstr(
            Cjob[i] + M * (1 - eta[k, i, last]) >= Theta[k + 1, m_last]
        )
    model.addConstr(Cmax >= Cjob[i])

# -----------------------------
# Objective: weighted sum
# -----------------------------
w1 = 1.0   # weight for makespan
w2 = 0.1   # weight for sum of job completion times
w3 = 0.05  # weight for waiting times

obj = (
    w1 * Cmax
    + w2 * gp.quicksum(Cjob[i] for i in jobs)
    + w3 * gp.quicksum(W[i, t] for i in jobs for t in range(len(routes[i]) - 1))
)

model.setObjective(obj, GRB.MINIMIZE)

# -----------------------------
# Solve
# -----------------------------
model.Params.OutputFlag = 1
model.optimize()

# -----------------------------
# Print solution
# -----------------------------
if model.status == GRB.OPTIMAL:
    print("\nOptimal solution found:")
    print(f"Objective = {model.objVal:.3f}, Cmax = {Cmax.X:.3f}\n")

    print("Schedule (stage -> operations):")
    for k in range(S):
        scheduled = []
        for i in jobs:
            for t in range(len(routes[i])):
                if eta[k, i, t].X > 0.5:
                    scheduled.append((i, t, routes[i][t][0], routes[i][t][1]))
        if scheduled:
            print(f" Stage {k+1}: ", end="")
            print(", ".join(f"Job{job}-Op{op+1} (M{m}, p={p})"
                            for job, op, m, p in scheduled))

    print("\nTheta table:")
    for k in range(S+1):
        for m in machines:
            print(f" Theta[{k},{m}] = {Theta[k,m].X:.1f}")

    print("\nWaiting W[i,t]:")
    for i in jobs:
        for t in range(len(routes[i]) - 1):
            print(f" W[{i},{t}] = {W[i,t].X:.3f}")

    print("\nJob completion Cjob[i]:")
    for i in jobs:
        print(f" Cjob[{i}] = {Cjob[i].X:.3f}")
else:
    print("No optimal solution found. Status:", model.status)
