import numpy as np
from pod import *


class DMD:
    def __init__(self):

        self.singvals = None
        self.modes = None
        self.time = None

        self.dt = None
        self.t1 = None
        self.n_timesteps = None
        self.tikhonov = None
        self.x_cond = None

    def decompose(self,
                  X,
                  center=False,
                  alg="svd",
                  rank=0,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  Y=None,
                  dt=None,):

        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)

        self.n_timesteps = X.shape[1]
        self.pod_ = POD
        u, s, vh = self.pod_.decompose(
            X, alg, rank=rank, opt_trunc=opt_trunc, center=center)

        s_inv = np.zeros(s.shape)
        s_inv = 1 / s
        s_inv_ = s_inv.copy()
        if self.tikhonov:
            s_inv_ *= s**2 / (s**2 + self.tikhonov * self.x_cond)
        store = np.linalg.multi_dot((Y, vh.T, np.diag(s_inv_)))
        self.A_tilde = u.T @ store

        lambd, w = np.linalg.eig(self.A_tilde)
        if sorting == "abs":
            idx = (np.abs(lambd)).argsort()[::-1]
        else:
            idx = (np.real(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        phi = store @ w

        self.singvals = s
        self.modes = u
        self.time = vh

        self.dt = dt
        omega = np.log(lambd) / dt

        self.dmd_modes = phi
        self.lambd = lambd
        self.eigenvalues = omega

        return u, s, vh

    def predict(self, t, init=0, t1=0, method=0):

        if method:
            b, _, _, _ = np.linalg.lstsq(self.dmd_modes, init, rcond=None)
            b /= np.exp(self.eigenvalues * t1)
        else:
            alpha1 = self.singvals * self.time[:, 0]
            b = np.linalg.solve(self.lambd * self.low_dim_eig, alpha1) / np.exp(
                self.eigenvalues * t1
            )

        return self.dmd_modes @ (np.exp(np.outer(self.eigenvalues, t).T) * b).T

    def reconstruct(self, rank):

        t = np.linspace(self.t1, self.t1 + (self.n_timesteps - 1)
                        * self.dt, self.n_timesteps)
        y0 = np.linalg.multi_dot((self.modes[:, :rank], np.diag(self.singvals[:rank]), self.time[:rank, 0]))
        return np.hstack((y0, self.predict(t, t1=self.t1)))
