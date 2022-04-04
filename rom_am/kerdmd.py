import numpy as np
from numpy import s_
from .dmd import DMD
from .pod import POD
import warnings


class KERDMD(DMD):

    def decompose(self, X, alg="snap", rank=0, opt_trunc=False, tikhonov=0, sorting="abs", Y=None, dt=None, kernel="poly", p=2, a=1):

        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]

        # Computing X.T @ Y/X using chosen kernels
        if kernel == "poly":
            XTY = (1 + X.T @ Y)**p
            XTX = (1 + X.T @ X)**p
        elif kernel == "sigmoid":
            XTY = np.tanh(X.T @ Y + a)
            XTX = np.tanh(X.T @ X + a)
        elif kernel == "gaussian":
            XTY = np.exp(-X.T @ Y)
            XTX = XTY

        # Eigendecomposition of X.T @ Y to compute singular values and time coefficients
        vals, v = np.linalg.eigh(XTX)

        vals = np.flip(vals)
        v = np.flip(v, 1)
        vals[vals < 1e-10] = 0
        s = np.sqrt(vals)

        if opt_trunc:
            if X.shape[0] <= X.shape[1]:
                beta = X.shape[0]/X.shape[1]
            else:
                beta = X.shape[1]/X.shape[0]
            omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
            tau = np.median(s) * omega
            rank = np.sum(s > tau)
        else:
            if rank == 0:
                rank = len(vals[vals > 1e-10])
            elif 0 < rank < 1:
                rank = np.searchsorted(
                    np.cumsum(s**2 / (s**2).sum()), rank) + 1

        s = s[:rank]
        s_inv = np.zeros(s.shape)
        s_inv[s > 1e-10] = 1.0 / s[s > 1e-10]
        v = v[:, :rank]
        vh = v.T

        # Computing the Koopman operator approximation
        self._A = np.linalg.multi_dot(
            (np.diag(s_inv), vh, XTY, vh.T, np.diag(s_inv)))

        lambd, w = np.linalg.eig(self.A)
        if sorting == "abs":
            idx = (np.abs(lambd)).argsort()[::-1]
        else:
            idx = (np.real(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        # Computing the high-dimensional DMD modes [1]
        phi = np.linalg.multi_dot((X, vh.T, np.diag(s_inv), w))
        omega = np.log(lambd) / dt  # Continuous system eigenvalues

        # Loading the DMD instance's attributes
        self.dt = dt
        self.singvals = s
        self.time = vh
        self.dmd_modes = phi
        self.lambd = lambd
        self.eigenvalues = omega
        self.modes = None

        return 0, s, vh
