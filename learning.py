from dp import *


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
    Y = 5
    
    f = np.load('files/f.npy')[:, :Y, :Y]
    y_true = np.load('files/y.npy')
    w = np.load('files/w.npy')[:2 * Y]
    b = np.load('files/b.npy')[:2 * Y]
    features = np.load('files/features.npy')

    n = f.shape[0]
    Y = f.shape[1]
    
    rvces = []
    losses = []
    
    optim = AdamW(weight_decay=0.0001)
    
    for i in range(500):
        
        # G, s, t = create_graph(f)
    
        loss, y_pred = evaluate_loss(f, y_true)
        
        # loss += weight_decay / 2 * np.sum(w ** 2)
        
        # w, b = update_params_sgd(features, w, b, y_true, y_pred, weight_decay=0)
        
        dw, db = calc_grads(features, w, b, y_true, y_pred)
        w, b = optim.step(i + 1, w, b, dw, db)
    
        f = recalculate_f(features, w, b)
        
        rvce = abs(y_pred.sum() - y_true.sum()) / y_true.sum()
        
        print(f'i: {i} | loss: {loss:.2f} | c: {y_pred.sum()} | rvce: {rvce:.2f} | weights: {np.sum(w ** 2)}')
        
        rvces.append(rvce)
        losses.append(loss)
        
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('margin rescaling loss')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('margin rescaling loss')
    axes[0].plot(losses)
    
    axes[1].set_title('rvce')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('rvce')
    
    axes[1].plot(rvces)
    plt.savefig('outputs/plot_adamw_0.0001.png')