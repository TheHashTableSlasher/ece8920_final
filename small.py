#!/usr/bin/env python3

import collections
import csv
import itertools
from pprint import pp

import numpy as np
from scipy.optimize import linprog

Node = collections.namedtuple("Node", ["id", "lon", "lat", "outbound", "inbound"])

graph = {}
edge_idxs = {}
c = []

with open("nodes_small.csv", newline="") as nodesfile, open("edges_small.csv", newline="") as edgesfile:
    for row in csv.DictReader(nodesfile):
        graph[row["id"]] = Node(row["id"], float(row["lon"]), float(row["lat"]), [], [])
    
    for k, row in enumerate(csv.DictReader(edgesfile)):
        graph[row["from"]].outbound.append((row["to"], float(row["weight"])))
        graph[row["to"]].inbound.append((row["from"], float(row["weight"])))
        edge_idxs[(row["from"], row["to"])] = k
        c.append(float(row["weight"]))

depots = collections.Counter(["Tacoma", "Tacoma", "Pullman"])
customers = [k for k in graph.keys() if k not in depots]

A = []
b = []
Aeq = []
beq = []

for v in customers:
    # Condition 1
    row = np.zeros(len(c))
    
    for u, weight in graph[v].outbound:
        row[edge_idxs[(v, u)]] = 1
        
    Aeq.append(row)
    beq.append(1)
    
    # Condition 2
    row = np.zeros(len(c))
    
    for u, weight in graph[v].inbound:
        row[edge_idxs[(u, v)]] = 1
        
    Aeq.append(row)
    beq.append(1)

for v, k in depots.items():
    # Condition 3
    row = np.zeros(len(c))
    
    for u, weight in graph[v].outbound:
        row[edge_idxs[(v, u)]] = 1
        
    Aeq.append(row)
    beq.append(k)
    
    # Condition 4
    row = np.zeros(len(c))
    
    for u, weight in graph[v].inbound:
        row[edge_idxs[(u, v)]] = 1
        
    Aeq.append(row)
    beq.append(k)

powerset = itertools.chain.from_iterable(itertools.combinations(customers, r) for r in range(1, len(customers) + 1))

for r in powerset:
    # Condition 5
    r = set(r)
    
    row = np.zeros(len(c))
    modified = False
    
    for (u, v), i in edge_idxs.items():
        if u not in r and v in r:
            row[i] = -1
            modified = True
    
    if modified:
        A.append(row)
        b.append(-1)

c = np.array(c)
A = np.array(A)
b = np.array(b)
Aeq = np.array(Aeq)
beq = np.array(beq)

print("c = ", c)
print("A = ", A)
print("b = ", b)
print("Aeq = ", Aeq)
print("beq = ", beq)

x = linprog(c, A, b, Aeq, beq, (0, 1), integrality=np.ones(len(c)))  
print("x = ", [u + " -> " + v for (u, v), i in edge_idxs.items() if x.x[i] > 0.5])  
