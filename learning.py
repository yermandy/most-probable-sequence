from dp import *
import pickle


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
            b = b - self.weight_decay * b
            
        return w, b


if __name__ == '__main__':
    Y = 6
    
    root = 'files'
    
    w = np.load(f'{root}/w.npy')[:2 * Y]
    b = np.load(f'{root}/b.npy')[:2 * Y]
    
    with open(f'{root}/y_trn_val.pickle', 'rb') as f:
        y_true = pickle.load(f)
    
    with open(f'{root}/features_trn_val.pickle', 'rb') as f:
        features = pickle.load(f)
    
    with open(f'{root}/y_tst.pickle', 'rb') as f:
        y_true_tst = pickle.load(f)
    
    with open(f'{root}/features_tst.pickle', 'rb') as f:
        features_tst = pickle.load(f)
    
    # print(f.shape)
    # print(y_true.shape)

    # n = f.shape[0]
    # Y = f.shape[1]
    
    rvces_epochs = []
    losses_epochs = []
    
    rvces_epochs_tst = []
    losses_epochs_tst = []
    
    
    optim = AdamW(weight_decay=0.0001)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for i in range(500):
        
        # G, s, t = create_graph(f)
        
        dw = 0
        db = 0
        
        rvces = []
        losses = []
        
        losses_tst = []
        rvces_tst = []
        
        for features_i, y_true_i in zip(features, y_true):
        
            f = calc_f(features_i, w, b)
        
            loss, y_pred = evaluate_loss(f, y_true_i)
            losses.append(loss)
            
            rvce = abs(y_pred.sum() - y_true_i.sum()) / y_true_i.sum()
            rvces.append(rvce)
            
            dw_i, db_i = calc_grads(features_i, w, b, y_true_i, y_pred)
            
            dw += dw_i
            db += db_i
                        
        dw /= len(features)
        db /= len(features)
            
        w, b = optim.step(i + 1, w, b, dw, db)
        
        for features_i, y_true_i in zip(features_tst, y_true_tst):
            f = calc_f(features_i, w, b)
        
            loss, y_pred = evaluate_loss(f, y_true_i)
            losses_tst.append(loss)
            
            rvce = abs(y_pred.sum() - y_true_i.sum()) / y_true_i.sum()
            rvces_tst.append(rvce)

        rvce = np.mean(rvces)
        loss = np.mean(losses)
        
        print(f'i: {i} | loss: {loss:.2f} | rvce: {rvce:.2f} | weights: {np.sum(w ** 2)}')
        
        rvces_epochs.append(rvce)
        losses_epochs.append(loss)
        
        rvces_epochs_tst.append(np.mean(rvces_tst))
        losses_epochs_tst.append(np.mean(losses_tst))
        
        axes[0].cla()
        axes[0].set_title('margin rescaling loss')
        axes[0].set_xlabel('iteration')
        axes[0].set_ylabel('margin rescaling loss')
        axes[0].plot(losses_epochs, label='trn, val')
        axes[0].plot(losses_epochs_tst, label='tst')
        
        axes[1].cla()
        axes[1].set_title('rvce')
        axes[1].set_xlabel('iteration')
        axes[1].set_ylabel('rvce')
        axes[1].plot(rvces_epochs, label='trn, val')
        axes[1].plot(rvces_epochs_tst, label='tst')
        
        for ax in axes: ax.legend()
    
        plt.tight_layout()
        plt.pause(.0001)
        
        
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].set_title('margin rescaling loss')
    # axes[0].set_xlabel('iteration')
    # axes[0].set_ylabel('margin rescaling loss')
    # axes[0].plot(losses_epochs)
    
    # axes[1].set_title('rvce')
    # axes[1].set_xlabel('iteration')
    # axes[1].set_ylabel('rvce')
    # axes[1].plot(rvces_epochs)
    
    plt.savefig('outputs/plot_trn_val.png')