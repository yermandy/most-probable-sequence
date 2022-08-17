from dp import *
import wandb
import os
from arguments import parse_args
from most_probable_sequence import most_probable_sequence
from utils import *
from optim import AdamW, SGD
from rich import print
from tqdm.rich import tqdm


def batch(N, batch_size):
    indices = np.arange(N)
    np.random.shuffle(indices)
    for i in range(0, N, batch_size):
        yield np.array(indices[i : i + batch_size])


def inference(features, y_true, w, b, calculate_loss=False):
    losses = []
    rvces = []

    for features_i, y_true_i in zip(features, y_true):
        f = calc_f(features_i, w, b)
    
        if calculate_loss:
            loss, _ = evaluate_loss(f, y_true_i)
            losses.append(loss)

        _, y_pred = most_probable_sequence(f)
        
        rvce = abs(y_pred.sum() - y_true_i.sum()) / y_true_i.sum()
        rvces.append(rvce)

    mean_loss = np.mean(losses) if calculate_loss else 0
    mean_rvce = np.mean(rvces)

    return mean_loss, mean_rvce


def load_params(run_name):
    w = np.load(f'outputs/{run_name}/w.npy')
    b = np.load(f'outputs/{run_name}/b.npy')
    return w, b


def set_seed(seed):
    np.random.seed(seed)


def get_optim(optim, lr, weight_decay):
    if optim == 'AdamW':
        return AdamW(lr=lr, weight_decay=weight_decay)
    elif optim == 'SGD':
        return SGD(lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer {optim}')


def filter_data(y, features):
    y_filtered = []
    features_filtered = []
    for y_i, features_i in zip(y, features):
        if y_i.sum() > 0:
            y_filtered.append(y_i)
            features_filtered.append(features_i)
    return y_filtered, features_filtered


if __name__ == '__main__':
    os.environ['WANDB_MODE'] = 'disabled'

    args = parse_args()
    print(args)

    run = wandb.init(project="most-probable-sequence", entity="yermandy")
    run_name = wandb.run.name
    
    root = 'files/031_more_validation_samples'
    split = args.split
    
    w = np.load(f'{root}/params/split_{split}/w.npy')[:2 * args.Y]
    b = np.load(f'{root}/params/split_{split}/b.npy')[:2 * args.Y]

    # trn_folder = ['files/trn/shuffled/5_minutes/10_samples', 'files/trn/shuffled/10_minutes/10_samples']
    # trn_folder = [f'{root}/trn/split_{split}/shuffled/10_minutes/5_samples', f'{root}/trn/split_{split}/shuffled/whole_file']
    
    trn_folder = [f'{root}/trn/split_{split}/shuffled/whole_file']
    val_folder = f'{root}/val/split_{split}/shuffled/whole_file'
    tst_folder = f'{root}/tst/split_{split}/shuffled/whole_file'
    
    y_true_trn, features_trn = get_data(trn_folder)
    y_true_trn, features_trn = filter_data(y_true_trn, features_trn)

    y_true_val, features_val = get_data(val_folder)
    y_true_val, features_val = filter_data(y_true_val, features_val)
    
    y_true_tst, features_tst = get_data(tst_folder)

    set_seed(args.seed)

    if args.cross_validation_fold >= 0:
        y_true_all = y_true_trn + y_true_val
        features_all = features_trn + features_val

        from sklearn.model_selection import KFold
        folds = KFold(n_splits=5, random_state=args.seed, shuffle=True).split(features_all)

        trn_indices, val_indices = list(folds)[args.cross_validation_fold]
        
        y_true_val = [y_true_all[i] for i in val_indices]
        features_val = [features_all[i] for i in val_indices]

        y_true_trn = [y_true_all[i] for i in trn_indices]
        features_trn = [features_all[i] for i in trn_indices]

        del y_true_all, features_all
    
    optim = get_optim(args.optim, args.lr, args.weight_decay)

    loss_val_best = np.inf
    rvce_val_best = np.inf

    wandb.config.update(args)
    wandb.config.update({
        'n_trn_samples': len(y_true_trn),
        'n_val_samples': len(y_true_val),
        'n_tst_samples': len(y_true_tst),
        'trn_folder': trn_folder,
        'val_folder': val_folder,
        'tst_folder': tst_folder,
        'normalization_in_loss': True
    })

    os.makedirs(f'outputs/{run_name}')

    loss_tst, rvce_tst = inference(features_tst, y_true_tst, w, b, calculate_loss=True)
    loss_val, rvce_val = inference(features_val, y_true_val, w, b, calculate_loss=True)

    wandb.log({
        'initial tst loss': loss_tst,
        'initial tst rvce': rvce_tst,
        'initial val loss': loss_val,
        'initial val rvce': rvce_val
    })

    for i in tqdm(range(args.epochs)):
        loss_trn = []

        for indices in batch(len(features_trn), args.batch_size):
            dw = []
            db = []

            for idx in indices:
                features_i = features_trn[idx]
                y_true_i = y_true_trn[idx]
            
                f = calc_f(features_i, w, b)
            
                loss, y_tilde = evaluate_loss(f, y_true_i)
                loss_trn.append(loss)
                
                dw_i, db_i = calc_grads(features_i, w, b, y_true_i, y_tilde)

                dw.append(dw_i)
                db.append(db_i)

            dw = np.mean(dw, axis=0)
            db = np.mean(db, axis=0)

            if args.biases_only:
                dw = 0

            w, b = optim.step(i + 1, w, b, dw, db)

        loss_trn = np.mean(loss_trn)
        _, rvce_trn = inference(features_trn, y_true_trn, w, b)
        
        print(f'trn | i: {i} | loss: {loss_trn:.2f} | rvce: {rvce_trn:.2f}')

        log = {
            'trn loss': loss_trn,
            'trn rvce': rvce_trn,
            'weights': np.sum(w ** 2),
            'biases': np.sum(b ** 2)
        }

        if args.validation:

            loss_val, rvce_val = inference(features_val, y_true_val, w, b)

            if loss_val <= loss_val_best and rvce_val <= rvce_val_best:
                loss_val_best = loss_val
                rvce_val_best = rvce_val

                np.save(f'outputs/{run_name}/w.npy', w)
                np.save(f'outputs/{run_name}/b.npy', b)

            log.update({
                'val loss': loss_val,
                'val rvce': rvce_val,
                'val rvce best': rvce_val_best,
                'val loss best': loss_val_best
            })

            print(f'val | i: {i} | loss: {loss_val:.2f} | rvce: {rvce_val:.2f}')

        if args.testing:

            loss_tst, rvce_tst = inference(features_tst, y_true_tst, w, b)
            
            log.update({
                'tst loss': loss_tst,
                'tst rvce': rvce_tst
            })

            print(f'tst | i: {i} | loss: {loss_tst:.2f} | rvce: {rvce_tst:.2f}')

        wandb.log(log)

    w, b = load_params(run_name)
    loss_tst, rvce_tst = inference(features_tst, y_true_tst, w, b, calculate_loss=True)
    loss_val, rvce_val = inference(features_val, y_true_val, w, b, calculate_loss=True)

    wandb.log({
        'final tst loss': loss_tst,
        'final tst rvce': rvce_tst,
        'final val loss': loss_val,
        'final val rvce': rvce_val
    })

    run.finish()