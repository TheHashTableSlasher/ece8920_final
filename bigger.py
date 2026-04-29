#!/usr/bin/env python3

import collections
import csv
import itertools
import sys
from pprint import pp

import numpy as np
from scipy.sparse import csr_array
from scipy.optimize import linprog

np.set_printoptions(threshold=sys.maxsize)

Node = collections.namedtuple("Node", ["id", "lon", "lat", "outbound", "inbound"])

graph = {}
c = []

drivers = [
    ("Tacoma", (lambda u, v, c: c * (0.5 if v.lat < graph["Tacoma"].lat and v.lon < graph["Moses Lake"].lon else 1.0))),
    ("Tacoma", (lambda u, v, c: c * (0.5 if v.lat > graph["Tacoma"].lat and v.lon < graph["Moses Lake"].lon else 1.0))),
    ("Pullman", (lambda u, v, c: c + (1000.0 if v.lon < graph["Moses Lake"].lon else 0.0)))
]

#drivers = [
#    ("Tacoma", (lambda u, v, c: c)),
#    ("Tacoma", (lambda u, v, c: c)),
#    ("Pullman", (lambda u, v, c: c))
#]

with open("nodes_small.csv", newline="") as nodesfile, open("edges_small.csv", newline="") as edgesfile:
    for row in csv.DictReader(nodesfile):
        graph[row["id"]] = Node(row["id"], float(row["lon"]), float(row["lat"]), [], [])
    
    for row in csv.DictReader(edgesfile):
        graph[row["from"]].outbound.append((row["to"], float(row["weight"])))
        graph[row["to"]].inbound.append((row["from"], float(row["weight"])))
        c.append((graph[row["from"]], graph[row["to"]], float(row["weight"])))

edge_idxs = {}
c_ = []

for i, (depot, transform) in enumerate(drivers):
    for u, v, weight in c:
        edge_idxs[(i, u.id, v.id)] = len(c_)
        c_.append(transform(u, v, weight))

depots = {driver[0] for driver in drivers}
customers = [k for k in graph.keys() if k not in depots]

c = c_ + [0] * len(drivers) * len(customers)

A = []
b = []
Aeq = []
beq = []

for ii, v in enumerate(customers):
    condvars = range(len(edge_idxs) + ii * len(drivers), len(edge_idxs) + (ii + 1) * len(drivers))

    for j, (i, _) in zip(condvars, enumerate(drivers)):
        # Condition 1
        row = np.zeros(len(c))
        
        for u, weight in graph[v].outbound:
            row[edge_idxs[(i, v, u)]] = 1
        
        row[j] = -1
            
        Aeq.append(row)
        beq.append(0)
        
        # Condition 2
        row = np.zeros(len(c))
        
        for u, weight in graph[v].inbound:
            row[edge_idxs[(i, u, v)]] = 1
        
        row[j] = -1
            
        Aeq.append(row)
        beq.append(0)
        
    # NEW CONDITION: Ensure each vertex is only accounted for by one vehicle
    row = np.zeros(len(c))
    
    for j in condvars:
        row[j] = 1
        
    Aeq.append(row)
    beq.append(1)

for i, (v, _) in enumerate(drivers):
    # Condition 3
    row = np.zeros(len(c))
    
    for u, _ in graph[v].outbound:
        row[edge_idxs[(i, v, u)]] = 1
        
    Aeq.append(row)
    beq.append(1)
    
    # Condition 4
    row = np.zeros(len(c))
    
    for u, _ in graph[v].inbound:
        row[edge_idxs[(i, u, v)]] = 1
        
    Aeq.append(row)
    beq.append(1)


powerset = itertools.chain.from_iterable(itertools.combinations(customers, r) for r in range(1, len(customers) + 1))

for r in powerset:
    # Condition 5
    r = set(r)
    
    row = np.zeros(len(c))
    modified = False
    
    for (i, u, v), j in edge_idxs.items():
        if u not in r and v in r:
            row[j] = -1
            modified = True
    
    if modified:
        A.append(row)
        b.append(-1)
        
c = np.array(c)
A = np.array(A)
b = np.array(b)
Aeq = np.array(Aeq)
beq = np.array(beq)

print(A.shape)
print(Aeq.shape)

print("c =\n", c)
print("Aeq =\n", Aeq)
print("beq =\n", beq)

x = linprog(c, A, b, Aeq, beq, (0, 1), integrality=np.ones(len(c)))
pp(x)

paths = [[] for _ in range(len(drivers))]

for (i, u, v), j in edge_idxs.items():
    if x.x[j] > 0.5:
        paths[i].append(u + " -> " + v)
            
print("x = ", paths)
