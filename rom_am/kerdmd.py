import numpy as np
from .dmd import DMD
from .pod import POD
import warnings


class KERDMD(DMD):

    def decompose(self, X, alg="svd", rank=0, opt_trunc=False, tikhonov=0, sorting="abs", Y=None, dt=None, kernel="Poly"):

        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]

        # Computing X.T @ Y using chosen kernels
        if kernel == "Poly":
            XTY = (1 + X.T @ Y)**2

        # Eigendecomposition of X.T @ Y to compute singular values and time coefficients
        times, s = np.linalg.eig(XTY)

        # Computing the Koopman operator approximation
        self._A = np.linalg.multi_dot(
            (np.diag(s), times.T, XTY, times, np.diag(1/s)))

        lambd, w = np.linalg.eig(self.A)
        if sorting == "abs":
            idx = (np.abs(lambd)).argsort()[::-1]
        else:
            idx = (np.real(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        # Computing the high-dimensional DMD modes [1]
        phi = w
        omega = np.log(lambd) / dt  # Continuous system eigenvalues

        # Loading the DMD instance's attributes
        self.dt = dt
        self.singvals = s
        self.time = times
        self.dmd_modes = phi
        self.lambd = lambd
        self.eigenvalues = omega
        self.modes = None
