import numpy as np
from tabulate import tabulate


def generate_random(seed=42):
    np.random.seed(seed)
    
    f = np.random.rand(4, 3, 3)
    f[0, 1, 0] = 2
    f[1, 0, 2] = 2
    f[2, 2, 2] = 2
    f[3, 2, 1] = 2
    
    f[0, 0, 2] = 3
    f[1, 2, 1] = 1
    f[2, 1, 0] = 3
    f[3, 0, 2] = 2
    
    f[0, 2, 1] = 3
    f[1, 1, 0] = 3
    f[2, 0, 1] = 3
    f[3, 1, 0] = 3

    y = np.array([2, 1, 0, 1, 0])
    
    return f, y


def generate_random_from_params():
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
    b = np.random.rand(2 * Y)
    # features ϕ
    phi = np.random.rand(d, n)

    # f_i(y_i = j, y_{i+1} = k)
    f = np.zeros((n, Y, Y))
    for i in range(n):
        outputs = w @ phi[:, i] + b
        for j in range(Y):
            for k in range(Y):
                f[i, j, k] = outputs[j + k]
    
    return f


def print_dict_as_table(dictionary: dict):
    table = [] 
    for k, v in dictionary.items():
        table.append([k, v])
    print(tabulate(table))