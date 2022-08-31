import numpy as np


class AdamW:
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

            # According to "Deep Learning" by Goodfellow, biases should not be regularized, see p.230
            # db = db + self.weight_decay * b

        # Update biased first moment estimate
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        # Update biased second raw moment estimate
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * dw**2
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * db**2

        # Compute bias-corrected first moment estimate
        m_dw_corr = self.m_dw / (1 - self.beta1**t)
        m_db_corr = self.m_db / (1 - self.beta1**t)

        # Compute bias-corrected second raw moment estimate
        v_dw_corr = self.v_dw / (1 - self.beta2**t)
        v_db_corr = self.v_db / (1 - self.beta2**t)

        # Update parameters
        w = w - self.lr * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)
        b = b - self.lr * m_db_corr / (np.sqrt(v_db_corr) + self.epsilon)

        if self.weight_decay > 0:
            w = w - self.lr * self.weight_decay * w

            # According to "Deep Learning" by Goodfellow, biases should not be regularized, see p.230
            # b = b - lr * self.weight_decay * b

        return w, b


class SGD:
    def __init__(self, lr=0.001, weight_decay=0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, t, w, b, dw, db):
        # Update parameters
        w = w - self.lr * dw
        b = b - self.lr * db

        if self.weight_decay > 0:
            w = w - self.lr * self.weight_decay * w

            # According to "Deep Learning" by Goodfellow, biases should not be regularized, see p.230
            # b = b - lr * self.weight_decay * b

        return w, b
