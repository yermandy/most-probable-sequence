from collections import defaultdict
import wandb
import pickle
import numpy as np
from inference import evaluate, evaluate_map


api = wandb.Api()

rvces = []
rvces_map = []

for split in [0, 1, 2, 3, 4]:
    # D = defaultdict(list)
    # d = defaultdict(list)
    predictions = defaultdict(list)

    tst_files_root = 'files/031_RX100_resized_128_sr_22050'

    with open(f'{tst_files_root}/tst/split_{split}/shuffled/whole_file/y.pickle', 'rb') as f:
        y_true = pickle.load(f)

    with open(f'{tst_files_root}/tst/split_{split}/shuffled/whole_file/features.pickle', 'rb') as f:
        features = pickle.load(f)

    for inner_split in range(5):

        runs = api.runs("yermandy/most-probable-sequence",filters={
                            "$and": [
                                {"config.cross_validation_fold": inner_split},
                                {"config.split": split},
                        ]},
                        order='summary_metrics.val rvce best')

        # print(runs)
        # print(len(runs))
        for run in runs:
            if 'cross_validation_fold' in run.config:
                # print(run.config['cross_validation_fold'])
                # print(split, run.name)
                # print(run.config['lr'])
                # print(run.config['weight_decay'])
                run_name = run.name
                lr = run.config['lr']
                wd = run.config['weight_decay']
                # val_rvce_best = run.summary['final tst rvce']
                val_rvce_best = run.summary['val rvce best']

                print(split, inner_split, run.name, val_rvce_best)
                
                # D[(lr, wd)].append(val_rvce_best)
                # print(run.summary)

                weights_root = f'{tst_files_root}/params/split_{split}'
                rvces_run = evaluate(features, y_true, weights_root, run_name, predictions=predictions)

                # print(rvces_run)
                break
                
    # print(predictions)
    for i, y_i in enumerate(y_true):
        pred_i = np.array(predictions[i])
        # pred_i = pred_i.mean(0).sum()
        pred_i = np.median(pred_i, axis=0).sum()
        true_i = y_i.sum()

        rvce = abs(pred_i.sum() - true_i.sum()) / true_i.sum()
        rvces.append(rvce)

    rvce_map = evaluate_map(features, y_true, weights_root)
    rvces_map.extend(rvce_map)

        
print('STRUCTURED')
print(f'{np.mean(rvces):.3f} ± {np.std(rvces):.3f}')

print('MAP')
print(f'{np.mean(rvces_map):.3f} ± {np.std(rvces_map):.3f}')