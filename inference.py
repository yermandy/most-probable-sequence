from dp import *
import pickle

from most_probable_sequence import most_probable_sequence


root = 'files'

with open(f'{root}/tst/y.pickle', 'rb') as f:
    y_true = pickle.load(f)
    
with open(f'{root}/tst/features.pickle', 'rb') as f:
    features = pickle.load(f)


def load_weights(run_name):
    if run_name != None:
        w = np.load(f'outputs/{run_name}/w.npy')
        b = np.load(f'outputs/{run_name}/b.npy')
    else:
        w = np.load(f'{root}/w.npy')
        b = np.load(f'{root}/b.npy')

    return w, b


def evaluate_map():
    w, b = load_weights(None)

    rvces = []

    for features_i, y_true_i in zip(features, y_true):
        features_i = features_i[::2]
        scores = w @ features_i.T + b.reshape(-1, 1)
        y_pred = scores.argmax(0)
        rvce = abs(y_pred.sum() - y_true_i.sum()) / y_true_i.sum()
        rvces.append(rvce)
        # print('rvce:', rvce, ' | c_pred:', y_pred.sum(), ' | c_true:', y_true_i.sum())

    print('MAP')
    print(f'RVCE: {np.mean(rvces)} : {np.std(rvces)}')
    print()


def evaluate(run_name=None):
    w, b = load_weights(run_name)
    
    Y = 6
    w = w[:2 * Y]
    b = b[:2 * Y]

    losses = []
    rvces = []

    for features_i, y_true_i in zip(features, y_true):

        f = calc_f(features_i, w, b)
                    
        # loss, y_pred = evaluate_loss(f, y_true_i)
        length, y_pred = most_probable_sequence(f)
        # losses.append(loss)

        rvce = abs(y_pred.sum() - y_true_i.sum()) / y_true_i.sum()
        rvces.append(rvce)

    print('Structured')
    print(run_name if run_name != None else 'initial')
    # print(f'Loss: {np.mean(losses)} : {np.std(losses)}')
    print(f'RVCE: {np.mean(rvces)} : {np.std(rvces)}')
    print()


runs = [
    'resilient-fire-308',
    'lemon-tree-298',
    'honest-valley-303',
    'charmed-snowball-290'
]

evaluate_map()

evaluate()

for run_name in runs:
    evaluate(run_name)