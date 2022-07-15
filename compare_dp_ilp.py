from dp import *
from utils import generate_random


for seed in range(0, 100):
    
    f, y = generate_random(seed)

    n = f.shape[0]
    Y = f.shape[1]

    C_max = (Y - 1) * (n + 1)

    F, Is = dymanic_programming(f, n, Y)
    G, s, t = create_graph(f)

    for c in range(1, C_max + 2):
        objective_ilp, maximizers_ilp = evaluate(f, G, s, t, c - 1)
        objective_dp, maximizers_dp = backtrack(Is, F, c, Y)
        
        print(c - 1, objective_dp, maximizers_dp, '<- DP')
        print(c - 1, *evaluate(f, G, s, t, c - 1), '<- ILP')
        print()
        
        assert objective_dp - objective_ilp < 1e-6
        assert np.all(maximizers_ilp == maximizers_dp)