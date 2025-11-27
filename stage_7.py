"""
para_sdm_optB.py  -- Option B: Aggressive optimization (reachable nodes)

Strategy:
 - For each job, build reachable nodes per stage k by following transition rules
   (BFS-like expansion). Γ_k contains union of reachable nodes across jobs.
 - Create eta variables only for transitions that appear in the reachable expansion.
 - Build disjunctions only for arcs that exist.
 - Same variable names (theta, eta, y, diff, z).
"""

import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib.pyplot as plt
import random

# -------------------------
# Example job data (replace with your table)
# -------------------------
jobs = {
    1: [(1, 8),  (2, 5),  (3, 7),  (4, 6)],
    2: [(2, 6),  (3, 9),  (4, 5),  (1, 7)],
    3: [(3, 7),  (4, 4),  (1, 6),  (2, 8)],
    4: [(4, 5),  (1, 9),  (2, 7),  (3, 6)],
}
machines = sorted({m for seq in jobs.values() for m,_ in seq})
K = sum(len(seq) for seq in jobs.values())

# -------------------------
# Transition successor function (local)
# -------------------------
def successors_for(nf, j, seq):
    """
    Return list of nt nodes that nf can go to for job j given seq.
    This enumerates only the small set of possible successors per the rules.
    """
    succ = []
    if nf[0] == "J":
        succ.extend([("MJ", j, 1), ("WJ", j, 1)])
        return succ
    if nf[0] == "MJ":
        _, r = nf[1], nf[2]
        succ.append(("WJ", j, r))
        if r < len(seq):
            succ.append(("MJ", j, r+1)); succ.append(("WJ", j, r+1))
        else:
            succ.append(("Cj", j))
        return succ
    if nf[0] == "WJ":
        jn, r = nf[1], nf[2]
        succ.append(("WJ", j, r)); succ.append(("MJ", j, r))
        return succ
    if nf[0] == "Cj":
        succ.append(("Cj", j)); succ.append(("delta",))
        return succ
    if nf[0] == "delta":
        succ.append(("delta",)); return succ
    # fallback
    return succ

# -------------------------
# Build minimal reachable nodes per stage (Option B)
# -------------------------
def build_nodes_optB(jobs, K):
    # nodes_by_stage is dict stage->set(nodes)
    nodes_by_stage = {k:set() for k in range(K+2)}
    # initialize stage 0
    for j in jobs:
        nodes_by_stage[0].add(("J", j))
    # expand stage-wise using successors_for
    for k in range(0, K):
        # for each job, consider nodes at stage k reachable for that job
        for j, seq in jobs.items():
            gamma0 = ("J", j)
            # restrict nf to nodes at stage k that either belong to job j or are global placeholders
            for nf in list(nodes_by_stage[k]):
                # We permit transitions only if nf is relevant to job j (J/j, MJ/WJ/Cj for same job)
                if nf[0] in ("J","delta"):
                    # J and delta are fine
                    pass
                elif nf[0] in ("MJ","WJ","Cj"):
                    # node belongs to some job; if different job, skip
                    if nf[1] != j:
                        continue
                elif nf[0] == "M":
                    # machine placeholder (we keep machine placeholders as possible targets)
                    pass
                else:
                    continue
                # get successors per rules
                for nt in successors_for(nf, j, seq):
                    nodes_by_stage[k+1].add(nt)
        # also optionally include machine placeholders only if some MJ at next stage uses them
        # but we will include M,m in all stages where any MJ exists using machine m
        # we will add M nodes after union stage built
    # Now add machine placeholders M,m to stages where they are needed
    for k in range(1, K+1):
        used_machines = set()
        # if any MJ in nodes_by_stage[k] uses machine m, include ("M",m)
        for node in list(nodes_by_stage[k]):
            if node[0] == "MJ":
                j, r = node[1], node[2]
                m = jobs[j][r-1][0]
                used_machines.add(m)
        for m in used_machines:
            nodes_by_stage[k].add(("M", m))
        # always include delta
        nodes_by_stage[k].add(("delta",))
    nodes_by_stage[K+1] = {("delta",)}
    # convert sets to sorted lists for deterministic ordering
    nodes_list = {k: sorted(nodes_by_stage[k]) for k in nodes_by_stage}
    return nodes_list

nodes = build_nodes_optB(jobs, K)

# -------------------------
# same helper functions
# -------------------------
def proc_time_of(node):
    if node[0] == "MJ":
        j, r = node[1], node[2]
        return jobs[j][r-1][1]
    return 0.0

def machine_of(node):
    if node[0] == "MJ":
        j, r = node[1], node[2]
        return jobs[j][r-1][0]
    return None

def feasible_transition(j0, nf, nt, seq):
    # same exact rules as before - used to create eta only if allowed
    if nf[0] == "J":
        j = nf[1]
        if j != j0:
            return False
        return nt in [("MJ", j, 1), ("WJ", j, 1)]
    if nf[0] == "MJ":
        j, r = nf[1], nf[2]
        if nt == ("WJ", j, r):
            return True
        if r < len(seq):
            return nt in [("MJ", j, r+1), ("WJ", j, r+1)]
        return nt == ("Cj", j)
    if nf[0] == "WJ":
        j, r = nf[1], nf[2]
        return nt in [("WJ", j, r), ("MJ", j, r)]
    if nf[0] == "Cj":
        j = nf[1]
        return nt in [("Cj", j), ("delta",)]
    if nf[0] == "delta":
        return nt == ("delta",)
    return False

# -------------------------
# Build model using reachable nodes only
# -------------------------
model = gp.Model("ParaSDM_optB")
model.Params.OutputFlag = 1
M = sum(p for seq in jobs.values() for _,p in seq) + 100

# theta
theta = {}
for k in nodes:
    for g in nodes[k]:
        theta[(k,g)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_{k}_{str(g).replace(' ','')}")

# eta only for reachable transitions
eta = {}
for k in range(0, K+1):
    for j0, seq in jobs.items():
        gamma0 = ("J", j0)
        for nf in nodes[k]:
            # skip nf that cannot belong to this job to avoid creating useless variables
            if nf[0] in ("MJ","WJ","Cj") and nf[1] != j0:
                continue
            for nt in nodes[k+1]:
                if feasible_transition(j0, nf, nt, seq):
                    eta[(k,nf,nt,gamma0)] = model.addVar(vtype=GRB.BINARY, name=f"eta_{k}({str(nt)}|{str(nf)},J{j0})")

# collect machine arcs
machine_arcs = []
for key,a_var in eta.items():
    k,nf,nt,g0 = key
    if nt[0]=="MJ":
        m = machine_of(nt)
        if m is not None:
            machine_arcs.append((key,m,k,nf,nt,a_var))

# y ordering variables only for actual conflicting pairs
y={}
for i in range(len(machine_arcs)):
    key_i,m_i,k_i,nf_i,nt_i,a_i = machine_arcs[i]
    for j in range(i+1,len(machine_arcs)):
        key_j,m_j,k_j,nf_j,nt_j,a_j = machine_arcs[j]
        if m_i==m_j and k_i==k_j:
            y[(i,j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_k{k_i}")
            model.addConstr(theta[(k_i+1, nt_i)] <= theta[(k_j, nf_j)] + M*(1 - y[(i,j)]) + M*(2 - a_i - a_j))
            model.addConstr(theta[(k_j+1, nt_j)] <= theta[(k_i, nf_i)] + M*(y[(i,j)]) + M*(2 - a_i - a_j))

# time propagation
for key,a_var in eta.items():
    k,nf,nt,g0 = key
    p = proc_time_of(nt)
    model.addConstr(theta[(k+1,nt)] >= theta[(k,nf)] + p - M*(1 - a_var))

# flow: start must pick one
for j0 in jobs:
    gamma0=("J",j0)
    outs=[eta_key for eta_key in eta if eta_key[0]==0 and eta_key[1]==gamma0 and eta_key[3]==gamma0]
    model.addConstr(quicksum(eta[o] for o in outs)==1)

# flow conservation
for k in range(1,K+1):
    for j0,seq in jobs.items():
        gamma0=("J",j0)
        for g in nodes[k]:
            # only consider g that could be relevant to j0 (skip MJ/WJ of other jobs)
            if g[0] in ("MJ","WJ","Cj") and g[1] != j0:
                continue
            infl=[eta[(k-1,nf,g,gamma0)] for nf in nodes[k-1] if (k-1,nf,g,gamma0) in eta]
            out=[eta[(k,g,nt,gamma0)] for nt in nodes[k+1] if (k,g,nt,gamma0) in eta]
            if infl or out:
                model.addConstr(quicksum(infl)==quicksum(out))

# reach delta once
for j0 in jobs:
    gamma0=("J",j0)
    arcs_to_delta=[eta[key] for key in eta if key[2]==("delta",) and key[3]==gamma0]
    model.addConstr(quicksum(arcs_to_delta)==1)

# objective: linearize diff and z as before
diff={}
z={}
obj=gp.LinExpr()
for key,a_var in eta.items():
    k,nf,nt,g0=key
    diff[key]=model.addVar(lb=0.0,vtype=GRB.CONTINUOUS,name=f"diff_{k}_{str(nf)}_to_{str(nt)}_J{g0[1]}")
    model.addConstr(diff[key] >= theta[(k+1,nt)] - theta[(k,nf)])
    z[key]=model.addVar(lb=0.0,vtype=GRB.CONTINUOUS,name=f"z_{k}_{str(nf)}_to_{str(nt)}_J{g0[1]}")
    model.addConstr(z[key] <= diff[key])
    model.addConstr(z[key] <= M * a_var)
    model.addConstr(z[key] >= diff[key] - M*(1 - a_var))
    p = proc_time_of(nt)
    if p>0:
        obj += p * a_var
    obj += z[key]

model.setObjective(obj, GRB.MINIMIZE)

# start times zero
for j0 in jobs:
    model.addConstr(theta[(0,("J",j0))]==0.0)

model.Params.TimeLimit = 120
model.optimize()

# reconstruct function (same idea)
def reconstruct(j0):
    cur=("J",j0); path=[]
    for k in range(0,K+1):
        found=False
        for nt in nodes[k+1]:
            key=(k,cur,nt,("J",j0))
            if key in eta and eta[key].X>0.5:
                path.append((k,cur,nt)); cur=nt; found=True; break
        if not found or cur==("delta",): break
    return path

ops=[]
if model.Status in (GRB.OPTIMAL,GRB.TIME_LIMIT,GRB.SUBOPTIMAL):
    for j0 in jobs:
        path=reconstruct(j0)
        for (k,nf,nt) in path:
            if nt[0]=="MJ":
                j,r=nt[1],nt[2]; m=machine_of(nt)
                ops.append({'machine':m,'job':j,'op':r,'start':theta[(k,nf)].X,'end':theta[(k+1,nt)].X})

ops=sorted(ops,key=lambda x:(x['machine'],x['start']))

if ops:
    machines_used=sorted({o['machine'] for o in ops})
    mach_y={m:i for i,m in enumerate(machines_used)}
    fig,ax=plt.subplots(figsize=(14,6))
    colors={}
    for o in ops:
        key=(o['job'],o['op'])
        if key not in colors:
            random.seed(hash(key)%999999); colors[key]=(0.4+random.random()*0.4,0.4+random.random()*0.4,0.4+random.random()*0.4)
    for o in ops:
        y=mach_y[o['machine']]; dur=o['end']-o['start']
        ax.broken_barh([(o['start'],dur)],(y-0.35,0.7),facecolor=colors[(o['job'],o['op'])],edgecolor='black',linewidth=1.3)
        ax.text(o['start']+dur*0.1,y,f"J{o['job']}-O{o['op']}",va='center',ha='left',fontsize=10,fontweight='bold',color='white')
    ax.set_yticks([mach_y[m] for m in machines_used])
    ax.set_yticklabels([f"Machine {m}" for m in machines_used],fontsize=11)
    ax.set_xlabel("Time",fontsize=12); ax.set_title("Merged operation timeline",fontsize=14,fontweight='bold')
    ax.grid(axis='x',linestyle='--',linewidth=0.7,alpha=0.4)
    plt.tight_layout(); plt.show()

print("Objective:", model.ObjVal)