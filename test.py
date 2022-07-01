import matplotlib.pyplot as plt
from ilp import *
from most_probable_sequence import *

def generate_random():
    np.random.seed(42)
    
    # f = np.random.rand(20, 3, 3)
    
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

    y = [1, 0, 2, 2, 1]
    
    return f, y

def backtrack(Is, idx, c):
    maximizers = [idx]
    for k in reversed(list(Is.keys())):
        c -= idx
        idx = Is[k][c, idx]
        maximizers.insert(0, idx)
    return np.array(maximizers)


def draw(G):
    lengths = nx.get_edge_attributes(G, 'length')
    lengths = {l: round(v, 3) for l, v in lengths.items()}

    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=lengths, label_pos=0.6)


f, y_true = generate_random()
# const = find_true_score(f, y_true)


# exit()

sequence = most_probable_sequence(f)
print(sequence)
exit() 

G, s, t = create_graph(f)

n = f.shape[0]
Y = f.shape[1]
C_max = (Y - 1) * (n + 1)

print('ILP')
# for c in range(C_max + 1):
for c in range(C_max):
    print(c, *evaluate(f, G, s, t, c))

print('\nDP')

n = f.shape[0]
Y = f.shape[1]

F, Is = dymanic_programming(f, n, Y)
    
for c in range(1, C_max + 1):
    values = [F[c - y_n, y_n] for y_n in range(0, min(Y, c))]
    idx = np.argmax(values)
    obj = values[idx]
    maximizers = backtrack(Is, idx, c)
    print(c - 1, obj, maximizers)
    # print(c - 1, *evaluate(f, G, s, t, c - 1))


draw(G)
# plt.show()