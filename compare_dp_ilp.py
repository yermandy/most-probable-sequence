from dp import *
from utils import generate_random


f, y = generate_random(0)

n = f.shape[0]
Y = f.shape[1]

C_max = (Y - 1) * (n + 1)

F, Is = dymanic_programming(f, n, Y)
G, s, t = create_graph(f)

for c in range(1, C_max + 2):
    objective, maximizers = backtrack(Is, F, c, Y)
    
    print(c - 1, objective, maximizers, '<- DP')
    print(c - 1, *evaluate(f, G, s, t, c - 1), '<- ILP')
    
    print()