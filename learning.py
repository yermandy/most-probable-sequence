from dp import *


if __name__ == '__main__':
    # f = create_f()
    
    f = np.load('files/f.npy')
    y_true = np.load('files/y.npy')
    w = np.load('files/w.npy')[:20]
    b = np.load('files/b.npy')[:20]
    features = np.load('files/features.npy')

    # print(len(f), len(y))
    # exit()
    
    n = f.shape[0]
    Y = f.shape[1]
            
    # c_best, obj_best = optimize_c(f)
    
    
    rvces = []
    losses = []
    
    for i in range(20):
        
        # G, s, t = create_graph(f)
    
        loss, y_pred = evaluate_loss(f, y_true)
        
        w, b = update_params(features, w, b, y_true, y_pred)
    
        f = recalculate_f(features, w, b)
        
        rvce = abs(y_pred.sum() - y_true.sum()) / y_true.sum()
        
        # rvce = np.random.rand(1)[0]
        # loss = np.random.rand(1)[0]
        
        print(f'i: {i} | loss: {loss:.2f} | rvce: {rvce:.2f}')
        
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
    # plt.savefig('outputs/plot.png')