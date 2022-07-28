from dp import *
import pickle
from collections import defaultdict
from most_probable_sequence import most_probable_sequence


def load_weights(run_name):
    if run_name != None:
        w = np.load(f'outputs/{run_name}/w.npy')
        b = np.load(f'outputs/{run_name}/b.npy')
    else:
        w = np.load(f'{weights_root}/w.npy')
        b = np.load(f'{weights_root}/b.npy')
    return w, b


def evaluate_map(d=defaultdict(list)):
    w, b = load_weights(None)

    rvces = []

    for features_i, y_true_i in zip(features, y_true):
        features_i = features_i[::2]
        scores = w @ features_i.T + b.reshape(-1, 1)
        y_pred = scores.argmax(0)
        
        rvce = abs(y_pred.sum() - y_true_i.sum()) / y_true_i.sum()
        rvces.append(rvce)
        # print('rvce:', rvce, ' | c_pred:', y_pred.sum(), ' | c_true:', y_true_i.sum())
        y_t = y_true_i[::2] + y_true_i[1::2]

        assert len(y_t) == len(y_pred)

        for p, t in zip(y_pred, y_t):
            d[t].append(p)

    print('MAP')
    print(f'RVCE: {np.mean(rvces)} : {np.std(rvces)}')
    print()
    
    return rvces


def evaluate(run_name=None, d=defaultdict(list)):
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

        y_p = y_pred[::2] + y_pred[1::2]
        y_t = y_true_i[::2] + y_true_i[1::2]

        for p, t in zip(y_p, y_t):
            d[t].append(p)

    print('Structured')
    print(run_name if run_name != None else 'initial')
    # print(f'Loss: {np.mean(losses)} : {np.std(losses)}')
    print(f'RVCE: {np.mean(rvces)} : {np.std(rvces)}')
    print()
    return rvces


def plot(d, d_map):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    X = []
    Y = []
    Y_map = []
    distribution = []

    d = {k: d[k] for k in sorted(d)}
    for label, preds in d.items():
        X.append(label)
        Y.append(np.mean(preds))
        preds_map = d_map[label]
        Y_map.append(np.mean(preds_map) if len(preds_map) > 0 else 0)
        distribution.append(len(preds))

    axes[0].set_xlabel('True class')
    axes[0].set_ylabel('Average Predicted class')
    axes[0].plot(X, Y, 'o-', label='structured')
    axes[0].plot(X, Y_map, 'o-', label='MAP')
    axes[0].plot(range(len(X)), 'o-', label='true')
    axes[0].grid()
    axes[0].legend()

    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Number of events')
    axes[1].grid()
    axes[1].plot(distribution, 'o-')
    
    plt.tight_layout()
    plt.savefig('outputs/true_vs_pred_class.png')


if __name__ == '__main__':
    runs = [
        'divine-darkness-365', # split_0
        'elated-lake-399', # split_1
        'peach-oath-408', # split_2
        'fresh-durian-435', # split_3
        'peach-armadillo-462' # split_4
    ]

    d_map = defaultdict(list)
    d = defaultdict(list)

    rvces = []
    rvces_map = []
    for i, run_name in enumerate(runs):

        with open(f'files/tst/split_{i}/y.pickle', 'rb') as f:
            y_true = pickle.load(f)
        
        with open(f'files/tst/split_{i}/features.pickle', 'rb') as f:
            features = pickle.load(f)

        weights_root = f'files/split_{i}'

        # evaluate using MAP inference
        rvces_run_map = evaluate_map(d_map)
        
        rvces_map.extend(rvces_run_map)

        # evaluate using most probable sequence (not trained)
        evaluate()

        # evaluate using most probable sequence (trained)
        rvces_run = evaluate(run_name, d)
        
        rvces.extend(rvces_run)
    
    rvces = np.array(rvces)
    
    print('Final')
    print(f'RVCE: {np.mean(rvces)} : {np.std(rvces)}')
    print(f'RVCE MAP: {np.mean(rvces_map)} : {np.std(rvces_map)}')
    print()

    plot(d, d_map)