import numpy as np
import networkx as nx
import gurobipy as g
import matplotlib.pyplot as plt


def create_graph(f):
    n = f.shape[0]
    Y = f.shape[1]
    
    G = nx.DiGraph()

    s = -1
    t = Y * (n + 1)
    
    G.add_node(s, layer=-1)

    for i in range(Y):
        G.add_edge(s, i, length=0, capacity=i)

    for i in range(n):
        for j in range(Y):
            for k in range(Y):
                length = f[i, j, k]
                G.add_node(i * Y + j, layer=i)
                G.add_edge(i * Y + j, (i + 1) * Y + k, length=length, capacity=k)
        
    for i in range(Y):
        G.add_node(Y * n + i, layer=n)
        G.add_edge(Y * n + i, t, length=0, capacity=0)
        
    G.add_node(t, layer=n + 1)
    
    return G, s, t


def evaluate(f, G, s, t, C = 0):
    n = f.shape[0]
    Y = f.shape[1]

    env = g.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    # defining the model and variables
    # m = g.Model()
    m = g.Model(env=env)
    x = g.tupledict()
    for (v1, v2) in G.edges:
        x[v1, v2] = m.addVar(vtype=g.GRB.BINARY, name=f"x_{v1}_{v2}")
        # x[v1, v2] = m.addVar(lb=0, ub=1, vtype=g.GRB.CONTINUOUS, name=f"x_{v1}_{v2}")
        
    # adding flow conservation constraints
    for v in G.nodes:
        # we need to skip the source (s) and the sink (t)
        if v not in [s, t]:
            # we collect the predecessor variables
            expr1 = g.quicksum(x[u, v] for u in G.predecessors(v))
            
            # we collect the successor variables 
            expr2 = g.quicksum(x[v, w] for w in G.successors(v))
            
            # we add the constraint
            m.addConstr(expr1 == expr2)

    # add constraint that only one edge to (t) is active
    expr = g.quicksum(x[u, t] for u in G.predecessors(t))
    m.addConstr(expr == 1)

    expr = g.quicksum(x[v1, v2] * c for (v1, v2, c) in G.edges.data("capacity"))
    m.addConstr(expr == C)

    objective = g.quicksum(x[v1, v2] * l for (v1, v2, l) in G.edges.data("length"))

    m.setObjective(objective, sense=g.GRB.MAXIMIZE)
    
    m.optimize()
    
    maximizers = {}
    for i in range(n):
        for j in range(Y):
            for k in range(Y):
                if x[i * Y + j, (i + 1) * Y + k].x > 0.5:
                    maximizers[i] = (j, k)
    
    # import itertools
    # maximizers = {i: (j, k) for i, j, k in itertools.product(range(n), range(Y), range(Y)) if x[i * Y + j, (i + 1) * Y + k].x > 0.5}
    
    maximizers_array = []
    for i in range(n):
        maximizers_array.append(maximizers[i][0])
    maximizers_array.append(maximizers[i][1])
    maximizers_array = np.array(maximizers_array)
    
    return m.objVal, maximizers_array


def draw(G):
    lengths = nx.get_edge_attributes(G, 'length')
    lengths = {l: round(v, 3) for l, v in lengths.items()}

    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=lengths, label_pos=0.6)


if __name__ == '__main__':
    from utils import generate_random
    
    f, y = generate_random()
    
    G, s, t = create_graph(f)
    
    n = f.shape[0]
    Y = f.shape[1]
    C_max = (Y - 1) * (n + 1)

    print('ILP')
    for c in range(C_max + 1):
        print(c, *evaluate(f, G, s, t, c))
    
    draw(G)
    plt.show()