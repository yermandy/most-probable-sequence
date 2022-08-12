from dp import *
import pickle
from collections import defaultdict
from most_probable_sequence import most_probable_sequence
from utils import get_data
from rich import print


def load_weights(run_name, weights_root):
    if run_name != None:
        w = np.load(f'outputs/{run_name}/w.npy')
        b = np.load(f'outputs/{run_name}/b.npy')
    else:
        w = np.load(f'{weights_root}/w.npy')
        b = np.load(f'{weights_root}/b.npy')
    return w, b


def evaluate_map(features, y_true, weights_root, d=defaultdict(list)):
    w, b = load_weights(None, weights_root)

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
    print(f'{np.mean(rvces):.3f} ± {np.std(rvces):.3f}')
    print()
    
    return rvces


def evaluate(features, y_true, weights_root,  run_name=None, d=defaultdict(list), predictions=defaultdict(list)):
    w, b = load_weights(run_name, weights_root)
    
    Y = 6
    w = w[:2 * Y]
    b = b[:2 * Y]

    losses = []
    rvces = []

    for i, (features_i, y_true_i) in enumerate(zip(features, y_true)):

        f = calc_f(features_i, w, b)
                    
        length, y_pred = most_probable_sequence(f)
        predictions[i].append(y_pred)

        rvce = abs(y_pred.sum() - y_true_i.sum()) / y_true_i.sum()
        rvces.append(rvce)

        y_p = y_pred[::2] + y_pred[1::2]
        y_t = y_true_i[::2] + y_true_i[1::2]

        for p, t in zip(y_p, y_t):
            d[t].append(p)

    print('Structured')
    print(run_name if run_name != None else 'initial')
    print(f'{np.mean(rvces):.3f} ± {np.std(rvces):.3f}')
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
    plt.savefig('outputs/bmrm_true_vs_pred_class.png')


if __name__ == '__main__':

    normalize_X = False

    # ''' old
    runs = [
        'divine-darkness-365', # split_0
        'elated-lake-399', # split_1
        'peach-oath-408', # split_2
        'fresh-durian-435', # split_3
        'peach-armadillo-462' # split_4
    ]
    # ''' 

    ''' 031_RX100_resized_128_sr_22050
    # trained on:
    # - files/031_RX100_resized_128_sr_22050/trn/split_*/shuffled/whole_file
    runs = [
        'solar-wind-634', # split_0
        'floral-gorge-654', # split_1
        'restful-glitter-698', # split_2
        'noble-firefly-711', # split_3
        'leafy-music-763' # split_4
    ]
    # ''' 

    ''' 035_RX100_resized_128_audio_image_augmentation_bs_256
    runs = [
        'glad-terrain-496', # split_0
        'noble-waterfall-518', # split_1
        'sage-dragon-541', # split_2
        'eternal-cloud-563', # split_3
        'dry-wildflower-574' # split_4
    ]
    # ''' 

    ''' 031_RX100_resized_128_sr_22050
    # trained on:
    # - "files/031_RX100_resized_128_sr_22050/trn/split_*/shuffled/10_minutes/5_samples"
    # - "files/031_RX100_resized_128_sr_22050/trn/split_*/shuffled/whole_file"
    runs = [
        'honest-paper-1025',
        'frosty-moon-1035',
        'magic-silence-1076',
        'graceful-eon-1089',
        'laced-elevator-1111'
    ]
    # '''

    ''' 031_RX100_resized_128_sr_22050
    # trained on:
    # - "files/031_RX100_resized_128_sr_22050/trn/split_*/shuffled/10_minutes/5_samples"
    runs = [
        'silver-disco-893',
        'gentle-surf-922',
        'ruby-jazz-944',
        'resilient-wind-963',
        'twilight-waterfall-1008'
    ]
    # '''

    ''' 031_RX100_resized_128_sr_22050
    # 031_RX100_resized_128_sr_22050
    # trained on:
    # - "files/031_more_validation_samples/trn/split_4/shuffled/whole_file"
    runs = [
        'different-sunset-1159',
        'swift-tree-1198',
        'likely-sky-1224',
        'sage-glitter-1231',
        'young-sea-1259'
    ]
    # '''

    ''' 031_more_validation_samples
    # trained on:
    # - "files/031_more_validation_samples/trn/split_4/shuffled/whole_file"
    runs = [
        'vital-wood-1302',
        'treasured-bird-1305',
        'genial-grass-1313',
        'still-voice-1315',
        'astral-deluge-1322'
    ]
    # '''

    ''' BMRM
    runs = [
        'dauntless-microwave-39',
        'fallen-breeze-39',
        'driven-eon-41',
        'leafy-sun-43',
        'vocal-dew-41'
    ]
    # '''

    ''' BMRM with normalized X
    runs = [
        'devout-waterfall-84',
        'kind-rain-88',
        'laced-sun-86',
        'serene-forest-87',
        'fallen-dream-84'
    ]
    normalize_X = True
    # '''

    ''' BMRM only biases
    runs = [
        'solar-field-203',
        'pretty-lion-203',
        'solar-shadow-203',
        'fast-morning-203',
        'happy-feather-207'
    ]
    # '''

    d_map = defaultdict(list)
    d = defaultdict(list)

    tst_files_root = 'files/031_RX100_resized_128_sr_22050'

    rvces = []
    rvces_map = []
    for split, run_name in enumerate(runs):

        print('-' * 10)
        print(f'Split: {split}     Run: {run_name}\n')

        Y, X = get_data(f'{tst_files_root}/tst/split_{split}/shuffled/whole_file', normalize_X=normalize_X)

        weights_root = f'{tst_files_root}/params/split_{split}'

        # evaluate using MAP inference
        rvces_run_map = evaluate_map(X, Y, weights_root, d_map)
        
        rvces_map.extend(rvces_run_map)

        # evaluate using most probable sequence (not trained)
        evaluate(X, Y, weights_root)

        # evaluate using most probable sequence (trained)
        rvces_run = evaluate(X, Y, weights_root, run_name, d)
        
        rvces.extend(rvces_run)
    
    rvces = np.array(rvces)
    
    print('-' * 10)
    print('Final')
    print(f'STRUCTURED = {np.mean(rvces):.3f} ± {np.std(rvces):.3f}')
    print(f'MAP        = {np.mean(rvces_map):.3f} ± {np.std(rvces_map):.3f}')
    print()

    plot(d, d_map)