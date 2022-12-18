import numpy as np
import numba
import matplotlib.pyplot as plt

try:
    from ilp import *
except ImportError as e:
    print(e)


def calculate_score(f, y):
    return np.sum([f[i, y[i], y[i + 1]] for i in range(len(f))])


@numba.jit(nopython=True)
def backtrack(Is: dict, F: np.array, c: int, Y: int):
    n = len(Is)
    C = F.shape[0]
    lb = max(0, c - C + 1)
    F = np.flipud(F)
    values = np.diag(F, k=c - C + 1)

    y = np.argmax(np.asarray(values))
    objective = values[y]

    y += lb
    maximizers = [y]

    for k in range(n + 1, 1, -1):
        c -= y
        y = Is[k][c, y]
        maximizers.insert(0, y)

    return objective, np.array(maximizers)


@numba.jit(nopython=True)
def optimal_c(F, c_true, Y, n):
    objective_best = -np.inf
    c_best = None
    C = F.shape[0]
    F = np.flipud(F)
    C_max = (Y - 1) * n
    for c in range(0, C_max + 1):
        score = np.max(np.diag(F, k=c - C + 1))
        rvce_loss = abs(c - c_true) / c_true
        objective = rvce_loss + score / n
        if objective > objective_best:
            objective_best = objective
            c_best = c

    return c_best, objective_best


def evaluate_loss(f, y_true):
    n = f.shape[0]
    Y = f.shape[1]

    F, Is = dymanic_programming(f, n, Y)

    c_best, objective_best = optimal_c(F, y_true.sum(), Y, n)

    true_score = calculate_score(f, y_true)

    margin_rescaling_loss = objective_best - true_score / n

    # G, s, t = create_graph(f)
    # objective, y_pred = evaluate(f, G, s, t, c_best)
    # print('ilp objective', objective)

    objective, y_tilde = backtrack(Is, F, c_best, Y)
    # print('dp objective', objective)

    return margin_rescaling_loss, y_tilde


@numba.jit(nopython=True)
def dymanic_programming(f: np.array, n: int, Y: int):
    Is = {}

    F_prev = np.zeros((Y, Y), dtype=np.float64)
    I = np.full((Y, Y), -1, dtype=np.int64)
    for c in range(0, Y):
        for y_k in range(0, Y):
            F_prev[c, y_k] = f[0, c, y_k]
            I[c, y_k] = c

    Is[2] = I

    C = Y
    for k in range(3, n + 2):
        C += Y - 1
        # assert C == (Y - 1) * (k - 1) + 1
        F = np.zeros((C, Y))
        I = np.full((C, Y), -1, dtype=np.int64)
        for c in range(0, C):
            for y_k in range(0, Y):
                V = -np.inf
                lb = max(0, c - C + Y)
                # assert c - C + Y == c - (Y - 1) * (k - 2)
                for y_k_1 in range(lb, min(Y, c + 1)):
                    V_new = F_prev[c - y_k_1, y_k_1] + f[k - 2, y_k_1, y_k]
                    if V_new > V:
                        V = V_new
                        F[c, y_k] = V
                        I[c, y_k] = y_k_1
        F_prev = F
        Is[k] = I

    return F, Is


@numba.jit(nopython=True)
def calc_grads(features, w, b, y_true, y_tilde):
    w_grad = np.zeros_like(w)
    b_grad = np.zeros_like(b)

    for i, f in enumerate(features):
        z_pred = y_tilde[i] + y_tilde[i + 1]
        z_true = y_true[i] + y_true[i + 1]

        w_grad[z_pred] += f
        w_grad[z_true] -= f

        b_grad[z_pred] += 1
        b_grad[z_true] -= 1

    # normalize
    n = len(y_true) - 1
    w_grad /= n
    b_grad /= n

    return w_grad, b_grad


def calc_f(features, w, b):
    n = len(features)
    Y = len(w) // 2

    f = np.zeros((n, Y, Y))
    scores = features @ w.T + b.reshape(1, -1)

    for i in range(n):
        for j in range(Y):
            for k in range(Y):
                f[i, j, k] = scores[i, j + k]
    return f


if __name__ == "__main__":
    from utils import generate_random

    f, y = generate_random()

    n = f.shape[0]
    Y_trn = f.shape[1]

    C_max = (Y_trn - 1) * n

    F, Is = dymanic_programming(f, n, Y_trn)

    print("DP")
    for c in range(0, C_max + 3):
        objective, maximizers = backtrack(Is, F, c, Y_trn)
        print(c, objective, maximizers)

        # ? assert that DP objective calculated correctly
        longest_path = calculate_score(f, maximizers)
        assert longest_path - objective < 1e-8, f"{longest_path} != {objective}"

        # ? assert that DP objective is feasible
        assert c == maximizers.sum(), f"{c} != {maximizers.sum()}"
