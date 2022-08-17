from bmrm.base import bmrm
from rich import print
import numpy as np
from utils import *
from learning import *
import wandb
import os


class MarginRescallingLoss:
    def __init__(self, w_shape, b_shape, X, Y, logger=None):
        self.w_shape = w_shape
        self.b_shape = b_shape
        self.wb_shape = (w_shape[0], w_shape[1] + 1) 
        self.X = X
        self.Y = Y
        self.logger = logger
        
    def __call__(self, W):
        L = []
        dw = []
        db = []

        w, b = self.get_params(W)
        
        for x, y in zip(self.X, self.Y):
            f = calc_f(x, w, b)
                
            loss, y_tilde = evaluate_loss(f, y)
            dw_i, db_i = calc_grads(x, w, b, y, y_tilde)

            L.append(loss)
            dw.append(dw_i)
            db.append(db_i)

        L = np.mean(L)
        dw = np.mean(dw, axis=0)
        db = np.mean(db, axis=0)

        self.logger.log({
            'trn loss': L
        })

        dW = np.hstack((dw, db.reshape(-1, 1)))
        dW = dW.flatten()

        return L, dW


    def get_trainable_params_count(self):
        return np.prod(self.wb_shape)


    def get_params(self, W):
        wb = W.reshape(self.wb_shape)
        w, b = wb[:, :-1], wb[:, -1]
        return w, b


class MarginRescallingLossBiasesOnly:
    def __init__(self, w, b_shape, X, Y, logger=None):
        self.w = w
        self.b_shape = b_shape
        self.X = X
        self.Y = Y
        self.logger = logger
        
    def __call__(self, W):
        L = []
        db = []
        b = W
        
        for x, y in zip(self.X, self.Y):
            f = calc_f(x, self.w, b)
                
            loss, y_tilde = evaluate_loss(f, y)
            _, db_i = calc_grads(x, self.w, b, y, y_tilde)

            L.append(loss)
            db.append(db_i)

        L = np.mean(L)
        db = np.mean(db, axis=0)

        self.logger.log({
            'trn loss': L
        })

        dW = db

        return L, dW


    def get_trainable_params_count(self):
        return np.prod(self.b_shape)


    def get_params(self, W):
        b = W
        return self.w, b


def learn(args):
    os.environ['WANDB_MODE'] = 'disabled'

    args.optim = 'BMRM'

    run = wandb.init(project="audio-bmrm", entity="yermandy")
    run_name = wandb.run.name
    
    trn_folder = f'{args.root}/trn/split_{args.split}/shuffled/whole_file'
    val_folder = f'{args.root}/val/split_{args.split}/shuffled/whole_file'
    tst_folder = f'{args.root}/tst/split_{args.split}/shuffled/whole_file'
    
    Y_trn, X_trn = get_data(trn_folder, normalize_X=args.normalize_X)
    Y_trn, X_trn = filter_data(Y_trn, X_trn)

    Y_val, X_val = get_data(val_folder, normalize_X=args.normalize_X)
    Y_val, X_val = filter_data(Y_val, X_val)

    if args.combine_trn_and_val:
        Y_trn += Y_val
        X_trn += X_val

    Y_tst, X_tst = get_data(tst_folder, normalize_X=args.normalize_X)

    set_seed(args.seed)

    loss_val_best = np.inf
    rvce_val_best = np.inf

    wandb.config.update(args)
    wandb.config.update({
        'n_trn_samples': len(Y_trn),
        'n_val_samples': len(Y_val),
        'n_tst_samples': len(Y_tst),
        'trn_folder': trn_folder,
        'val_folder': val_folder,
        'tst_folder': tst_folder,
    })

    w = np.load(f'{args.root}/params/split_{args.split}/w.npy')[:2 * args.Y]
    b = np.load(f'{args.root}/params/split_{args.split}/b.npy')[:2 * args.Y]

    loss_tst, rvce_tst = inference(X_tst, Y_tst, w, b, calculate_loss=True)
    loss_val, rvce_val = inference(X_val, Y_val, w, b, calculate_loss=True)

    wandb.log({
        'initial tst loss': loss_tst,
        'initial tst rvce': rvce_tst,
        'initial val loss': loss_val,
        'initial val rvce': rvce_val
    })

    if args.biases_only:
        loss = MarginRescallingLossBiasesOnly(w, b.shape, X_trn, Y_trn, wandb)    
    else:
        loss = MarginRescallingLoss(w.shape, b.shape, X_trn, Y_trn, wandb)

    W, stats = bmrm(loss, loss.get_trainable_params_count(), lmbda=args.reg, tol_rel=args.tol_rel)
    
    w, b = loss.get_params(W)

    loss_tst, rvce_tst = inference(X_tst, Y_tst, w, b, calculate_loss=True)
    loss_val, rvce_val = inference(X_val, Y_val, w, b, calculate_loss=True)
    
    
    outputs_folder = f'outputs/{run_name}' if args.outputs_folder == None else args.outputs_folder
    
    os.makedirs(outputs_folder, exist_ok=True)
    np.save(f'{outputs_folder}/w.npy', w)
    np.save(f'{outputs_folder}/b.npy', b)

    wandb.log({
        'final tst loss': loss_tst,
        'final tst rvce': rvce_tst,
        'final val loss': loss_val,
        'final val rvce': rvce_val
    })

    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    
    print(args)
    
    learn(args)