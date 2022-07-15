import numpy as np
import numba
import matplotlib.pyplot as plt

try:
    from ilp import *
except ImportError as e:
    print(e)

def find_true_score(f, y):
    return np.sum(f[i, y[i], y[i + 1]] for i in range(len(f)))


def backtrack(Is, F, c, Y):
    # n = len(Is)
    # C_max = (Y - 1) * (n + 1) + 1
    # lb = max(0, c - C_max + 1)
    
    lb = 0
    values = [F[c - y_n, y_n] for y_n in range(lb, min(Y, c))]
    
    y = np.argmax(values)
    objective = values[y]

    y += lb
    maximizers = [y]
    
    for k in reversed(list(Is.keys())):
        c -= y
        y = Is[k][c, y]
        maximizers.insert(0, y)

    return objective, np.array(maximizers)


@numba.jit(nopython=True)
def optimal_c(F, c_hat, Y, n):
    objective_best = -np.inf
    c_best = None
    score_best = None
    
    C_max = (Y - 1) * (n + 1)
    for c in range(1, C_max + 1):
        score = max([F[c - y_n, y_n] for y_n in range(0, min(Y, c))])
        rvce_loss = abs(c - c_hat) / c_hat
        objective = rvce_loss + score
        # print(c, objective)
        if objective > objective_best:
            objective_best = objective
            c_best = c - 1
            score_best = score
    
    return c_best, objective_best
    

def evaluate_loss(f, y_true):
    n = f.shape[0]
    Y = f.shape[1]
    
    F, Is = dymanic_programming(f, n, Y)
    
    c_best, objective_best = optimal_c(F, y_true.sum(), Y, n)
    
    true_score = find_true_score(f, y_true)
    
    margin_rescaling_loss = objective_best - true_score
    
    # Notice, c_best should be c_best = c - 1
    # G, s, t = create_graph(f)
    # objective, y_pred = evaluate(f, G, s, t, c_best)
    # print('ilp objective', objective)
    
    objective, y_pred = backtrack(Is, F, c_best + 1, Y)
    # print('dp objective', objective)
    
    return margin_rescaling_loss, y_pred



@numba.jit(nopython=True)
def dymanic_programming(f: np.array, n: int, Y: int):
    
    Fs = {}
    Is = {}
    
    C = Y
    for k in range(2, n + 2):
        C += Y - 1
        F = np.zeros((C, Y))
        I = np.full((C, Y), -1, dtype=np.int64)
        for c in range(0, C):
            for y_k in range(0, Y):
                V = -np.inf
                lb = 0
                for y_k_1 in range(lb, min(Y, c)):
                    if k == 2:
                        F[c, y_k] = f[k - 2, y_k_1, y_k]
                        I[c, y_k] = y_k_1
                    else:
                        if c - y_k_1 >= C - Y + 1:
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
    
    F, Is = dymanic_programming(f, n, Y)
    
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


def update_params_sgd(features, w, b, y_true, y_pred, lr=1e-5, weight_decay=0.0):
    w_grad, b_grad = calc_grads(features, w, b, y_true, y_pred)
    
    if weight_decay > 0:
        w_grad += weight_decay * w
        
    w = w - lr * w_grad
    b = b - lr * b_grad
    
    return w, b


@numba.jit(nopython=True)
def calc_f(features, w, b):
    n = len(features)
    Y = len(w) // 2
    
    f = np.zeros((n, Y, Y))
    for i in range(n):
        scores = w @ features[i] + b

        for j in range(Y):
            for k in range(Y):
                f[i, j, k] = scores[j + k]
    return f


if __name__ == '__main__':
    from utils import generate_random
    
    f, y = generate_random()
    
    n = f.shape[0]
    Y = f.shape[1]
    
    # TODO: Should be
    # C_max = (Y - 1) * (n + 1)
    # but last iteration is not working
    
    C_max = (Y - 1) * n

    F, Is = dymanic_programming(f, n, Y)
        
    print('DP')
    for c in range(1, C_max + 1):
        objective, maximizers = backtrack(Is, F, c, Y)
        print(c - 1, objective, maximizers)
