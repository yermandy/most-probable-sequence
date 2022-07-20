from dp import *
from utils import generate_random


for seed in range(0, 100):
    
    f, y = generate_random(seed)

    n = f.shape[0]
    Y = f.shape[1]

    C_max = (Y - 1) * (n)

    F, Is = dymanic_programming(f, n, Y)
    G, s, t = create_graph(f)
    
    print('seed:', seed)

    for c in range(0, C_max + 1):
        objective_dp, maximizers_dp = backtrack(Is, F, c, Y)
        objective_ilp, maximizers_ilp = evaluate(f, G, s, t, c)
        
        print(c, objective_dp, maximizers_dp, '<- DP')
        print(c, objective_ilp, maximizers_ilp, '<- ILP')
        print()
        
        # ? assert that DP objective calculated correctly
        longest_path = find_true_score(f, maximizers_dp)
        assert longest_path - objective_dp < 1e-8, f'{longest_path} != {objective_dp}'
        
        # ? assert that DP objective is feasible
        assert c == maximizers_dp.sum(), f'{c} != {maximizers_dp.sum()}'
        
        # ! ILP objective can be lower than DP objective
        if objective_dp <= objective_ilp:
            
            # ? assert that DP maximizers are equal to ILP maximizers
            assert np.all(maximizers_ilp == maximizers_dp)
            
            # ? assert that DP objective is equal to ILP objective
            assert objective_dp - objective_ilp < 1e-8, f'{objective_dp} != {objective_ilp}'