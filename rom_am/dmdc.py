import numpy as np
from .pod import *

class DMDc:
    def decompose(self,
        X,
        center=False,
        alg="svd",
        rank=0,
        opt_trunc=False,
        tikhonov=0,
        sorting="abs",
        Y=None,
        dt=None,
        Y_input=None,):

        omega = np.vstack((X, Y_input))

        self.pod_til = POD
        self.pod_hat = POD
        u_til, s_til, vh_til = self.pod_til.decompose(omega, alg, rank=rank, opt_trunc=opt_trunc, center = center)
        u_til_1 = u_til[: X.shape[0], :]
        u_til_2 = u_til[: Y_input.shape[0], :]
        u_hat, _, _ = self.pod_hat.decompose(Y, alg, rank=rank, opt_trunc=opt_trunc, center = center)

        store_ = np.linalg.multi_dot((Y, vh_til.T, np.diag(1/s_til)))
        store = np.linalg.multi_dot((u_hat.T, store_))
        self.A_tilde = np.linalg.multi_dot((store, u_til_1.T, u_hat))
        self.B_tilde = np.linalg.multi_dot((store, u_til_2.T))

        lambd, w = np.linalg.eig(self.A_tilde)
        idx = np.abs(np.imag(lambd)).argsort()
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        phi = np.linalg.multi_dot((store_, u_til_1.T, u_hat, w))

        u = u_til_1
        vh = vh_til
        s = s_til

        omega = np.log(lambd) / dt

        self.dmd_modes = phi
        self.lambd = lambd
        self.eigenvalues = omega

        return u, s, vh
