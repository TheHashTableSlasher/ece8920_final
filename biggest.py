#!/usr/bin/env python3

import collections
from colorsys import hls_to_rgb
import csv
import heapq
import json
import itertools
import multiprocessing as mp
import re
import sys
import uuid

import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linprog
from sklearn.metrics.pairwise import haversine_distances

np.set_printoptions(threshold=sys.maxsize, linewidth=200)

customers = [
    [35.962558, -83.919284], 
    [35.914829, -84.082909], 
    [35.899309, -84.159375], 
    [35.955844, -83.934474], 
    [35.975818, -83.922276],
    [36.062214, -83.977709],
    [35.952024, -83.967558], 
    [35.959371, -83.915835], 
    [35.932104, -84.011989], 
    [36.044971, -84.005545], 
    [35.942404, -84.095447],
    [36.008022, -83.947809] 
]

depots = [
    [35.958559, -83.924930, lambda u, v, w: w * 1.5], 
    [36.079236, -83.946855, lambda u, v, w: w],
    [35.885915, -84.089886, lambda u, v, w: w]
]

npzfilename = sys.argv[1] if len(sys.argv) >= 2 else "biggest.npz"
geojsonfilename = sys.argv[2] if len(sys.argv) >= 3 else "biggest.geojson"

try:
    with open(npzfilename, "rb") as npzfile:
        npzdata = np.load(npzfile)
        
        X = npzdata["X"]
        A = npzdata["A"]
        b = npzdata["b"]
        Aeq = npzdata["Aeq"]
        beq = npzdata["beq"]
        z = npzdata["z"]
        customers = npzdata["customers"]
        depots = npzdata["depots"]
        
        @np.vectorize
        def dereference(k):
            k = bytes(k)
            if k == b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0":
                return None
            else:
                return npzdata[str(uuid.UUID(bytes=k))]
            
        geopaths = dereference(npzdata["geopaths"])
        
except FileNotFoundError:
    print("Building true graph...", end="", flush=True)

    G = {}
    
    Vertex = collections.namedtuple("Vertex", ["id", "lat", "lon", "edges"])
    Edge = collections.namedtuple("Edge", ["src", "dst", "weight", "path"])

    with open("nodes_big.csv", newline="") as nodefile, open("edges_big.csv", newline="") as edgesfile:
        for row in csv.DictReader(nodefile):
            id = int(row["id"])
            
            G[id] = Vertex(id, float(row["lat"]), float(row["lon"]), [])
        
        regex = re.compile(r"LINESTRING\(([-+]?(?:\d*\.*\d+) [-+]?(?:\d*\.*\d+)(, [-+]?(?:\d*\.*\d+) [-+]?(?:\d*\.*\d+))*)\)")
        
        for row in csv.DictReader(edgesfile):
            idsrc = int(row["from"])
            iddst = int(row["to"])
            
            if idsrc in G and iddst in G:
                regexmatch = regex.fullmatch(row["debug"])
                if regexmatch:
                    path = np.array([list(map(float, reversed(x.split()))) for x in regexmatch.group(1).split(",")])
                else:
                    path = np.array([[G[idsrc].lat, G[idsrc].lon], [G[iddst].lat, G[iddst].lon]])
                
                G[idsrc].edges.append(Edge(idsrc, iddst, float(row["weight"]), path))

    print(" OK")

    print("Finding nearest members to K_n in true graph...", end="", flush=True)

    ids = []
    kdt = []

    for v in G.values():
        ids.append(v.id)
        kdt.append([v.lat, v.lon])

    ids = np.array(ids)
    kdt = KDTree(kdt)

    def kdtsearch(point):
        for dist, i in zip(*kdt.query(point[:2], 100)):
            if G[ids[i]].edges:
                t = tuple([G[ids[i]]] + point[2:])
                return t[0] if len(t) == 1 else t
        raise RuntimeError()

    customers = list(map(kdtsearch, customers))
    depots = list(map(kdtsearch, depots))

    print(" OK")

    print("Building simplified K_n graph from true graph...", end="", flush=True)

    def astar(args):
        source, sink = args
        source, i = source
        sink, j = sink
        
        heap = [(0.0, source)]
        prev = collections.defaultdict(lambda: (float("inf"), None))
        
        while heap:
            wv, v = heapq.heappop(heap)
            
            if v == sink:
                break
            
            es = []
            
            for e in G[v].edges:
                if wv + e.weight < prev[e.dst][0]:
                    prev[e.dst] = (wv + e.weight, v)
                    es.append(e)
          
            if es:
                ws = np.array([e.weight for e in es])
                X = np.radians([[G[v].lat, G[v].lon]])
                Y = np.radians([[G[e.dst].lat, G[e.dst].lon] for e in es])
                ws = wv + ws + 6378100.0 * haversine_distances(X, Y).reshape(-1)
                
                for w, e in zip(ws, es):
                    heapq.heappush(heap, (w, e.dst))
                 
        path = [sink]
        
        while path[-1] != source:
            path.append(prev[path[-1]][1])
        
        geopath = [np.array([[G[source].lat, G[source].lon]])]
        
        for u, v in itertools.pairwise(reversed(path)):
            for e in G[u].edges:
                if e.dst == v:
                    if np.all(geopath[-1][-1, :] == e.path[0, :]):
                        geopath.append(e.path[1:, :])
                    else:
                        geopath.append(e.path)
                        
        last_path = np.array([[G[sink].lat, G[sink].lon]])
        if np.any(geopath[-1][-1, :] != last_path[0, :]):
            geopath.append(last_path)
        
        weights = [f(source, sink, prev[sink][0]) for v, f in depots]
        
        return (i, j, np.array(weights), np.concat(geopath))
        
    id2mat = {v.id: i for i, v in enumerate(itertools.chain(customers, (v for v, f in depots)))}
    X = np.empty((len(depots), len(id2mat), len(id2mat)))
    geopaths = np.empty((len(id2mat), len(id2mat)), dtype=object)

    with mp.Pool() as pool:
        for i, j, w, geopath in pool.imap_unordered(astar, itertools.permutations(id2mat.items(), 2)):
            X[:, i, j] = w
            geopaths[i, j] = geopath
    
    # This just makes things simpler for the v -> v case
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j, j] = 40075000.0
            
    print(" OK")
    
    customers_coords = np.array([[v.lat, v.lon] for v in customers], dtype=int)
    depots_coords = np.array([[v.lat, v.lon] for v in customers], dtype=int)
    customers = np.array([id2mat[v.id] for v in customers], dtype=int)
    depots = np.array([id2mat[v.id] for v, f in depots], dtype=int)
        
    print("Solving MDVRP...", end="", flush=True)

    c = np.append(X.reshape(-1), np.zeros(len(customers) * len(depots)))
    A = []
    b = []
    Aeq = []
    beq = []

    for i, x in enumerate(customers):
        for j, d in enumerate(depots):
            # Condition 1
            y = np.zeros(X.shape)
            z = np.zeros((len(customers), len(depots)))
            
            y[j, x, :] = 1
            y[j, x, x] = 0
            z[i, j] = -1
            
            Aeq.append(np.append(y.reshape(-1), z.reshape(-1)))
            beq.append(0)
            
            # Condition 2
            y = np.zeros(X.shape)
            z = np.zeros((len(customers), len(depots)))
            
            y[j, :, x] = 1
            y[j, x, x] = 0
            z[i, j] = -1
            
            Aeq.append(np.append(y.reshape(-1), z.reshape(-1)))
            beq.append(0)
        
        # Condition 1.5: ensure each vertex is only accounted for by one vehicle
        y = np.zeros(X.shape)
        z = np.zeros((len(customers), len(depots)))
        
        z[i, :] = 1
        
        Aeq.append(np.append(y.reshape(-1), z.reshape(-1)))
        beq.append(1)

    z = np.zeros(len(customers) * len(depots))

    for j, d in enumerate(depots):
        # Condition 3
        y = np.zeros(X.shape)
        
        y[j, d, :] = 1
        y[j, d, d] = 0
        
        Aeq.append(np.append(y.reshape(-1), z))
        beq.append(1)
        
        # Condition 4
        y = np.zeros(X.shape)
        
        y[j, :, d] = 1
        y[j, d, d] = 0
        
        Aeq.append(np.append(y.reshape(-1), z))
        beq.append(1)
        
        # Condition 4.5: only allow vehicles to use their specified depot
        y = np.zeros(X.shape)
        
        for e in depots:
            if e != d:
                y[j, e, :] = 1
                y[j, :, e] = 1
        
        Aeq.append(np.append(y.reshape(-1), z))
        beq.append(0)

    powerset = itertools.chain.from_iterable(itertools.combinations(customers, r) for r in range(1, len(customers) + 1))
    for s in powerset:
        # Condition 5: every subgraph is connected
        s = set(s)
        
        y = np.zeros(X.shape)
        
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                if i in s and j not in s:
                    y[:, i, j] = -1
                    
        A.append(np.append(y.reshape(-1), z))
        b.append(-1)

    A = np.array(A)
    b = np.array(b)
    Aeq = np.array(Aeq)
    beq = np.array(beq)

    z = linprog(c, A, b, Aeq, beq, (0, 1), integrality=np.ones(len(c)))
    z = z.x[:(np.prod(X.shape))].reshape(X.shape) > 0.5

    print(" OK")
        
    with open(npzfilename, "wb") as npzfile:
        _geopaths = np.zeros(geopaths.shape, dtype='|V16')
        
        tab = {
            "X": X, 
            "A": A,
            "b": b,
            "Aeq": Aeq,
            "beq": beq,
            "z": z,
            "customers": customers, 
            "depots": depots, 
            "geopaths": _geopaths
        }
        ns = uuid.uuid4()
        
        for (i, j), path in np.ndenumerate(geopaths):
            if path is not None:
                k = uuid.uuid5(ns, "{},{}".format(i, j))
                tab[str(k)] = path
                _geopaths[i, j] = k.bytes
       
        np.savez_compressed(npzfile, **tab)

features = []
path_colors = ["#{:02X}{:02X}{:02X}".format(*[int(x * 255.0) for x in hls_to_rgb(hue, 0.75, 1.0)]) for hue in np.linspace(0, 1, len(depots) + 1)[:len(depots)]]
customer_colors = "#729fcf"
depot_colors = "#ad7fa8"

# Add customers
features.append({
    "type": "Feature",
    "geometry": {
        "type": "MultiPoint",
        "coordinates": [[geopaths[c, depots[0]][0, 1], geopaths[c, depots[0]][0, 0]] for c in customers]
    },
    "properties": {
        "name": "Customers",
        "marker-color": customer_colors
    }
})

# Add Depots
features.append({
    "type": "Feature",
    "geometry": {
        "type": "MultiPoint",
        "coordinates": [[geopaths[d, customers[0]][0, 1], geopaths[d, customers[0]][0, 0]] for d in depots]
    },
    "properties": {
        "name": "Depots",
        "marker-color": depot_colors
    }
})

# Add paths
for ii, x in enumerate(z):
    lines = []
    
    for (i, j), tf in np.ndenumerate(x):
        if tf:
            lines.append(geopaths[i, j][:, ::-1].tolist())
    
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString", 
            "coordinates": lines
        },
        "properties": {
            "name": "Path of vehicle {}".format(ii + 1),
            "stroke": path_colors[ii]
        }
    })

with open(geojsonfilename, "w", encoding="utf-8") as geojsonfile:
    json.dump({"type": "FeatureCollection", "features": features}, geojsonfile)

print("MDVRP solution saved to " + geojsonfilename)
