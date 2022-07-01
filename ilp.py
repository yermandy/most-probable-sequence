#%%
import numpy as np
from tqdm import tqdm
import networkx as nx
import gurobipy as g
import numba
import matplotlib.pyplot as plt


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





def find_true_score(f, y):
    return np.sum([f[i, y[i], y[i + 1]] for i in range(len(f))])

    constant = 0
    for i in range(len(f)):
        y_i, y_i_1 = y[i], y[i + 1]
        constant += f[i, y_i, y_i_1]
        
    return constant
    

def evaluate_loss(f, G, s, t, y_true):
    n = f.shape[0]
    Y = f.shape[1]
    
    objective_best = -np.inf
    margin_rescaling_loss = None
    c_best = None
    score_best = None
    
    F = dymanic_programming(f, n, Y)
    
    c_hat = y_true.sum()
    for c in range(1, (Y - 1) * (n - 1) + 2):
        score = max([F[c - y_n, y_n] for y_n in range(0, min(Y, c))])
        rvce_loss = abs(c - c_hat) / c_hat
        objective = rvce_loss + score
        # print(c, objective)
        if objective > objective_best:
            objective_best = objective
            c_best = c - 1
            score_best = score

    true_score = find_true_score(f, y_true)
    
    margin_rescaling_loss = objective_best - true_score
    
    print(margin_rescaling_loss, score_best, c_best)
    
    objective, y_pred = evaluate(f, G, s, t, c_best)
    
    # print('ilp objective', objective)
    
    return margin_rescaling_loss, y_pred


# TODO there is still a bug somewhere (in the last iteration)
@numba.jit(nopython=True)
def dymanic_programming(f: np.array, n: int, Y: int):
    
    Fs = {}
    Is = {}
    
    for k in range(2, n + 2):
        C = Y * (k - 1) + 1
        # C = (Y - 1) * k + 1
        F = np.zeros((C, Y))
        I = np.full((C, Y), -1, dtype=np.int64)
        # print(k)
        for c in range(0, C):
            
            for y_k in range(0, Y):
                V = -np.inf
                for y_k_1 in range(0, min(Y, c)):
                    if k == 2:
                        F[c, y_k] = f[k - 2, y_k_1, y_k]
                        I[c, y_k] = y_k_1
                    else:
                        if c - y_k_1 >= Y * (k - 2) + 1:
                        # if c - y_k_1 >= (Y - 1) * (k - 1):
                            continue
                        
                        V_new = Fs[k - 1][c - y_k_1, y_k_1] + f[k - 2, y_k_1, y_k]
                        if V_new > V:
                            V = V_new
                            F[c, y_k] = V
                            I[c, y_k] = y_k_1
        Fs[k] = F
        Is[k] = I
    
    return F, Is


def optimize_c(f):
    n = f.shape[0]
    Y = f.shape[1]
    
    F = dymanic_programming(f, n, Y)
    
    obj_best = -np.inf
    c_best = None
    
    for c in range(1, (Y - 1) * (n - 1) + 2):
        values = [F[c - y_n, y_n] for y_n in range(0, min(Y, c))]
        obj = max(values)
        # print(c - 1, obj)
        if obj > obj_best:
            obj_best = obj
            c_best = c - 1
            
    print(c_best)
    print(obj_best)
            
    return c_best, obj_best
    

@numba.jit(nopython=True)
def calc_grads(features, w, b, y_true, y_pred):
    w_grad = np.zeros_like(w)
    b_grad = np.zeros_like(b)
    
    for i, f in enumerate(features):
        z_pred = y_pred[i] + y_pred[i + 1]
        z_true = y_true[i] + y_true[i + 1]
        
        w_grad[z_pred] += f
        w_grad[z_true] -= f
        
        b_grad[z_pred] += 1
        b_grad[z_true] -= 1
    
    return w_grad, b_grad


def update_params(features, w, b, y_true, y_pred, lr=5e-5):
    w_grad, b_grad = calc_grads(features, w, b, y_true, y_pred)
    
    w = w - lr * w_grad
    b = b - lr * b_grad
    
    return w, b
    
    
def recalculate_f(features, w, b):
    n = len(features)
    Y = int(len(w) / 2)
    
    f = np.zeros((n, Y, Y))
    for i in range(n):
        scores = w @ features[i] + b

        for j in range(Y):
            for k in range(Y):
                f[i, j, k] = scores[j + k]
    return f
    
    
# %%


if __name__ == '__main__':
    # f = create_f()
    
    f = load_f('f.npy')
    y_true = np.load('y.npy')
    w = np.load('w.npy')[:20]
    b = np.load('b.npy')[:20]
    features = np.load('features.npy')

    # print(len(f), len(y))
    # exit()
    
    n = f.shape[0]
    Y = f.shape[1]
            
    # c_best, obj_best = optimize_c(f)
    
    
    rvces = []
    losses = []
    
    for i in range(5):
        
        G, s, t = create_graph(f)
    
        loss, y_pred = evaluate_loss(f, G, s, t, y_true)
        
        w, b = update_params(features, w, b, y_true, y_pred)
    
        f = recalculate_f(features, w, b)
        
        rvce = abs(y_pred.sum() - y_true.sum()) / y_true.sum()
        
        # rvce = np.random.rand(1)[0]
        # loss = np.random.rand(1)[0]
        
        print(f'i: {i} | loss: {loss:.2f} | rvce: {rvce:.2f}')
        
        rvces.append(rvce)
        losses.append(loss)
        
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('margin rescaling loss')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('margin rescaling loss')
    axes[0].plot(losses)
    
    axes[1].set_title('rvce')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('rvce')
    
    axes[1].plot(rvces)
    # plt.savefig('plot.png')
        
        
    
    # print( np.sum((f_prime - f) > 5e-5 ))
    exit()
    objective, maximizers = evaluate(f, G, s, t, c_best)
    
    print()
    print(c_best)
    print(obj_best)
    print(objective)
    print(maximizers)
    
    print(len(maximizers))
    
    rvce = (maximizers.sum() - y_true.sum()) / y_true.sum()
    
    print(rvce)

    # for c in range((Y - 1) * (n - 1) + 2):
    # for c in range(100):
    #     objective, maximizers = evaluate(f, G, s, t, c)
    #     print(c, objective)  

    # len(y), y.sum()

    # sequence = most_probable_sequence(f)

    const = find_true_score(f, y_true)
    
    
# %%


# c_best: 85
# obj_best: 3592.455746650696
# objective: 3592.455746650696