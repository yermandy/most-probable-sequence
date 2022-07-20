from dp import *
import pickle
import wandb
import os
from arguments import args

class AdamW():
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        # Initialize 1st moment vector
        self.m_dw, self.m_db = 0, 0
        # Initialize 2nd moment vector
        self.v_dw, self.v_db = 0, 0
        # Exponential decay rates for the moment estimates
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
        self.weight_decay = weight_decay


    def step(self, t, w, b, dw, db):
        if self.weight_decay > 0:
            dw = dw + self.weight_decay * w
            #! bias decay?
            db = db + self.weight_decay * b
        
        # Update biased first moment estimate
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        # Update biased second raw moment estimate
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * dw ** 2
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * db ** 2

        # Compute bias-corrected first moment estimate
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        m_db_corr = self.m_db / (1 - self.beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        v_db_corr = self.v_db / (1 - self.beta2 ** t)

        # Update parameters
        w = w - self.lr * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)
        b = b - self.lr * m_db_corr / (np.sqrt(v_db_corr) + self.epsilon)
        
        if self.weight_decay > 0:
            w = w - self.weight_decay * w
            #! bias decay?
            b = b - self.weight_decay * b
            
        return w, b


class SGD():
    def __init__(self, lr=0.001, weight_decay=0):
        self.lr = lr
        self.weight_decay = weight_decay


    def step(self, t, w, b, dw, db):
        # Update parameters
        w = w - self.lr * dw
        b = b - self.lr * db

        if self.weight_decay > 0:
            w = w - self.weight_decay * w
            #! bias decay?
            b = b - self.weight_decay * b
            
        return w, b


def batch(N, batch_size):
    indices = np.arange(N)
    np.random.shuffle(indices)
    for i in range(0, N, batch_size):
        yield np.array(indices[i : i + batch_size])


def inference(features, y_true, w, b):
    losses = []
    rvces = []

    for features_i, y_true_i in zip(features, y_true):
        f = calc_f(features_i, w, b)
    
        loss, y_pred = evaluate_loss(f, y_true_i)

        losses.append(loss)
        
        rvce = abs(y_pred.sum() - y_true_i.sum()) / y_true_i.sum()
        rvces.append(rvce)

    mean_loss = np.mean(losses)
    mean_rvce = np.mean(rvces)

    return mean_loss, mean_rvce


def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def load_params(run_name):
    w = np.load(f'outputs/{run_name}/w.npy')
    b = np.load(f'outputs/{run_name}/b.npy')
    return w, b


def set_seed(seed):
    np.random.seed(seed)


def get_optim(optim_name, lr, weight_decay):
    if optim_name == 'AdamW':
        return AdamW(lr=lr, weight_decay=weight_decay)
    elif optim_name == 'SGD':
        return SGD(lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer {optim_name}')


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

    run = wandb.init(project="most-probable-sequence", entity="yermandy")

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    Y = args.Y
    seed = args.seed
    optim_name = args.optim

    validation = args.validation
    testing = args.testing

    run_name = wandb.run.name
    
    root = 'files'
    
    w = np.load(f'{root}/w.npy')[:2 * Y]
    b = np.load(f'{root}/b.npy')[:2 * Y]

    y_true_trn = load(f'{root}/y_trn_5_min.pickle')
    features_trn = load(f'{root}/features_trn_5_min.pickle')
    y_true_trn, features_trn = filter_data(y_true_trn, features_trn)

    y_true_val = load(f'{root}/y_val.pickle')
    features_val = load(f'{root}/features_val.pickle')
    y_true_val, features_val = filter_data(y_true_val, features_val)
    
    y_true_tst = load(f'{root}/y_tst.pickle')
    features_tst = load(f'{root}/features_tst.pickle')

    set_seed(seed)
    
    optim = get_optim(optim_name, lr, weight_decay)

    loss_val_best = np.inf
    rvce_val_best = np.inf

    wandb.config.update({
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'Y': Y,
        'validation': validation,
        'testing': testing,
        'seed': seed,
        'optim_name': optim_name,
        'n_trn_samples': len(y_true_trn),
        'n_val_samples': len(y_true_val),
        'sample_duration': 5,
    })

    os.makedirs(f'outputs/{run_name}')
    
    for i in range(epochs):                
        rvces_trn = []
        losses_trn = []

        for indices in batch(len(features_trn), batch_size):
            dw = []
            db = []

            for idx in indices:
                features_i = features_trn[idx]
                y_true_i = y_true_trn[idx]
            
                f = calc_f(features_i, w, b)
            
                loss, y_pred = evaluate_loss(f, y_true_i)
                losses_trn.append(loss)
                
                y_true_i_sum = y_true_i.sum()
                rvce = abs(y_pred.sum() - y_true_i_sum) / y_true_i_sum
                rvces_trn.append(rvce)
                
                dw_i, db_i = calc_grads(features_i, w, b, y_true_i, y_pred)

                dw.append(dw_i)
                db.append(db_i)

            dw = np.mean(dw, axis=0)
            db = np.mean(db, axis=0)

            w, b = optim.step(i + 1, w, b, dw, db)

        mean_rvce = np.mean(rvces_trn)
        mean_loss = np.mean(losses_trn)

        print(f'trn | i: {i} | loss: {mean_loss:.2f} | rvce: {mean_rvce:.2f}')

        log = {
            'trn loss': mean_loss,
            'trn rvce': mean_rvce,
            'weights': np.sum(w ** 2),
            'biases': np.sum(b ** 2)
        }

        if validation:

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

        if testing:

            loss_tst, rvce_tst = inference(features_tst, y_true_tst, w, b)
            
            log.update({
                'tst loss': loss_tst,
                'tst rvce': rvce_tst
            })

            print(f'tst | i: {i} | loss: {loss_tst:.2f} | rvce: {rvce_tst:.2f}')

        wandb.log(log)

    w, b = load_params(run_name)
    loss_tst, rvce_tst = inference(features_tst, y_true_tst, w, b)

    wandb.log({
        'final tst loss': loss_tst,
        'final tst rvce': rvce_tst
    })

    run.finish()