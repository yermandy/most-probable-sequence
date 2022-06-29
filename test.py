import matplotlib.pyplot as plt
from ilp import *

def generate_random():
    np.random.seed(42)
    
    # f = np.random.rand(6, 3, 3)
    
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


def draw(G):
    lengths = nx.get_edge_attributes(G, 'length')
    lengths = {l: round(v, 3) for l, v in lengths.items()}

    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=lengths, label_pos=0.6)


f, y = generate_random()
# const = find_constant(f, y)
# print(const)

# sequence = most_probable_sequence(f)
# print(sequence)
# exit() 

G, s, t = create_graph(f)


n = f.shape[0]
Y = f.shape[1]

# for c in range(Y * n + 2):
for c in range(0, Y * (n - 1) + 1):
    print(c, evaluate(f, G, s, t, c))

print('\nDP')

F = dymanic_programming(f)

for c in range(1, Y * (n - 1) + 1):
    values = [F[c - y_n, y_n] for y_n in range(0, min(Y, c))]
    obj = max(values)
    print(c - 1, obj)

draw(G)
# plt.show()