#%%
import numpy as np
from tqdm import tqdm
import networkx as nx
import gurobipy as g
import numba


def create_f():
    np.random.seed(42)

    # number of events in 3 second window
    Y = 10
    # dimension of feature vector
    d = 13
    # number of 3 second windows
    n = 50

    # <w(y_1 + y_2), ϕ(x_{12})>
    # weights w
    w = np.random.rand(2 * Y, d)
    # biases b
    b = np.random.rand(d)
    # features ϕ
    phi = np.random.rand(d, n)

    # f_i(y_i = j, y_{i+1} = k)
    f = np.zeros((n, Y, Y))
    for i in range(n):
        outputs = w @ phi[:, i]
        for j in range(Y):
            for k in range(Y):
                f[i, j, k] = outputs[j + k]
    return f


def load_f(path):
    return np.load(path)
    

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
    
    maximizers_array = []
    for i in range(n):
        maximizers_array.append(maximizers[i][0])
    maximizers_array.append(maximizers[i][1])
    maximizers_array = np.array(maximizers_array)
    
    return m.objVal, maximizers_array


def most_probable_sequence(f):
    f = np.copy(f)
    
    # find the most probable sequence using dynamic programming
    n = f.shape[0]
    Y = f.shape[1]

    I = np.zeros((Y, n), dtype=int)
    I[:, 0] = np.arange(Y)

    F = np.zeros((Y, n))

    for i in range(n):
        for k in range(Y):
            distances = f[i, :, k]
            if i > 0:
                distances += F[:, i - 1]
            # print(distances)
            maximizer = distances.argmax()

            I[k, i] = maximizer
            F[k, i] = distances[maximizer]

    idx = F[:, -1].argmax(0)
    length = F[idx, -1]
    print(length)

    sequence = []
    for i in reversed(range(n)):
        sequence.append(idx)
        idx = I[idx, i]
    sequence.append(idx)
    sequence = sequence[::-1]

    sequence = np.array(sequence)
    
    return sequence



# import matplotlib.pyplot as plt

# pos = nx.spring_layout(G, seed=0)  # Seed for reproducible layout
# nx.draw(G, pos, with_labels=True)





def find_constant(f, y):
    return np.sum([f[i, y[i], y[i + 1]] for i in range(len(f))])

    constant = 0
    for i in range(len(f)):
        y_i, y_i_1 = y[i], y[i + 1]
        constant += f[i, y_i, y_i_1]
        
    return constant
    

def evaluate_loss(f, G, s, t, y, const):
    n = f.shape[0]
    Y = f.shape[1]
    
    objective = -np.inf
    margin_rescaling_loss = None
    c_best = None
    
    c_hat = y.sum()
    # for c in range((Y - 1) * (n - 1) + 2):
    for c in range(200):
        F = evaluate(f, G, s, t, c)
        objective_new = abs(c - c_hat) / c_hat + F
        print(c, F)
        if objective_new > objective:
            objective = objective_new
            c_best = c

    margin_rescaling_loss = objective - const
    
    print(margin_rescaling_loss, c_best)
    
    return margin_rescaling_loss


@numba.jit
def dymanic_programming(f: np.array, n: int, Y: int):
    Fs = {}
    # Is = {}
    
    for k in range(2, n + 2):
        C = Y * (k - 1) + 1
        F = np.zeros((C, Y))
        # I = np.full((C, Y + 1), np.nan)
        print(k)
        for c in range(0, C):
            for y_k in range(0, Y):
                V = -np.inf
                for y_k_1 in range(0, min(Y, c)):
                    if k == 2:
                        F[c, y_k] = f[k - 2, y_k_1, y_k]
                        # I[c, y_k] = y_k_1
                    else:
                        if c - y_k_1 >= Y * (k - 2) + 1:
                            continue
                        
                        V_new = Fs[k - 1][c - y_k_1, y_k_1] + f[k - 2, y_k_1, y_k]
                        if V_new > V:
                            V = V_new
                            F[c, y_k] = V
                            # I[c, y_k] = y_k_1
        Fs[k] = F
        # Is[k] = I
    
    return F

    
# %%


if __name__ == '__main__':
    # f = create_f()
    
    f = load_f('f.npy')
    y = np.load('y.npy')
    
    n = f.shape[0]
    Y = f.shape[1]
    
    F = dymanic_programming(f, n, Y)
    
    n = f.shape[0]
    Y = f.shape[1]
    
    # for c in range(1, Y * (n - 1) + 1):
    
    obj_best = -np.inf
    c_best = None
    
    for c in range(1, (Y - 1) * (n - 1) + 2):
        values = [F[c - y_n, y_n] for y_n in range(0, min(Y, c))]
        obj = max(values)
        # print(c - 1, obj)
        if obj > obj_best:
            obj_best = obj
            c_best = c - 1
        
        
    
    G, s, t = create_graph(f)
    
    objective, maximizers = evaluate(f, G, s, t, c_best)
    
    print()
    print(c_best)
    print(obj_best)
    print(objective)
    print(maximizers)
    
    rvce = (maximizers.sum() - y.sum()) / y.sum()
    
    print(rvce)

    # for c in range((Y - 1) * (n - 1) + 2):
    # for c in range(100):
    #     objective, maximizers = evaluate(f, G, s, t, c)
    #     print(c, objective)  

    # len(y), y.sum()

    # sequence = most_probable_sequence(f)

    # const = find_constant(f, y)    
    
    # evaluate_loss(f, G, s, t, y, const)
# %%


# c_best: 85
# obj_best: 3592.455746650696
# objective: 3592.455746650696