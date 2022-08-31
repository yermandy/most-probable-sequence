import numpy as np
import jax.numpy as jnp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from bmrm.base import bmrm


class GeneralModel(BaseEstimator):
    def __init__(
        self,
        k=1,
        lmbda=1e-2,
        bias=True,
        solver="franc",
        tol_rel=1e-3,
        tol_abs=0,
        max_iter=np.inf,
        buff_size=500,
        cp_cln=np.inf,
        risk_transform=None,
    ):
        self.lmbda = lmbda
        self.bias = bias
        self.solver = solver
        self.tol_rel = tol_rel
        self.tol_abs = tol_abs
        self.max_iter = max_iter
        self.buff_size = buff_size
        self.cp_cln = cp_cln
        self.k = k
        self.risk_transform = risk_transform
        self.X = None
        self.y = None

    def fit(self, X, y, verb=False):
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
        if self.k % 1 > 0 or self.k < 1:
            raise ValueError("Model parameter k must be a positive integer")
        self.k = int(self.k)

        if self.bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        n_dim = X.shape[1] * self.k

        self.X = X
        self.y = y

        risk = self.risk
        if self.risk_transform is not None:
            risk = self.risk_transform(risk)

        self.W_, self.stats_ = bmrm(
            risk,
            n_dim,
            self.lmbda,
            self.solver,
            self.tol_rel,
            self.tol_abs,
            self.max_iter,
            self.buff_size,
            self.cp_cln,
            verb,
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return None

    def risk(self, W):
        return None


class BinaryClassifier(GeneralModel, ClassifierMixin):
    def __init__(
        self,
        lmbda=1e-2,
        bias=True,
        solver="franc",
        tol_rel=1e-3,
        tol_abs=0,
        max_iter=np.inf,
        buff_size=500,
        cp_cln=np.inf,
        risk_transform=None,
    ):
        super().__init__(
            lmbda=lmbda,
            bias=bias,
            solver=solver,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
            max_iter=max_iter,
            buff_size=buff_size,
            cp_cln=cp_cln,
            risk_transform=risk_transform,
        )

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        score = np.sign((self.W_ @ X.T))
        score[score == 0] = -1
        return score

    def risk(self, W):
        score = 1 - jnp.multiply((W @ self.X.T), self.y)
        R = jnp.where(score > 0, score, 0).mean()
        return R


class MultiClassifier(GeneralModel, ClassifierMixin):
    def __init__(
        self,
        k=2,
        lmbda=1e-2,
        bias=True,
        solver="franc",
        tol_rel=1e-3,
        tol_abs=0,
        max_iter=np.inf,
        buff_size=500,
        cp_cln=np.inf,
        risk_transform=None,
    ):
        super().__init__(
            k=k,
            lmbda=lmbda,
            bias=bias,
            solver=solver,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
            max_iter=max_iter,
            buff_size=buff_size,
            cp_cln=cp_cln,
            risk_transform=risk_transform,
        )

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        W = self.W_.reshape(self.k, X.shape[1])
        return np.argmax((W @ X.T), axis=0)

    def risk(self, W):
        W = W.reshape(-1, self.X.shape[1])
        wtx = W @ self.X.T
        witx = wtx[
            self.y.astype(int),
            jnp.linspace(0, self.y.shape[0] - 1, self.y.shape[0]).astype(int),
        ]
        brc = wtx - jnp.array(witx) + 1
        brc = brc.at[np.diag_indices(np.min(brc.shape))].set(0)
        return jnp.max(brc, axis=0).mean()


class Regressor(GeneralModel, RegressorMixin):
    def __init__(
        self,
        lmbda=1e-2,
        bias=True,
        solver="franc",
        tol_rel=1e-3,
        tol_abs=0,
        max_iter=np.inf,
        buff_size=500,
        cp_cln=np.inf,
        risk_transform=None,
    ):
        super().__init__(
            lmbda=lmbda,
            bias=bias,
            solver=solver,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
            max_iter=max_iter,
            buff_size=buff_size,
            cp_cln=cp_cln,
            risk_transform=risk_transform,
        )

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.W_ @ X.T

    def risk(self, W):
        return ((W @ self.X.T - self.y) ** 2).mean()
