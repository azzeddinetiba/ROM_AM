from msilib.schema import ODBCSourceAttribute
import numpy as np
from .dmd import DMD
from .pod import POD
import warnings


class EDMD(DMD):
    """
    Extended Dynamic Mode Decomposition Class

    """

    def decompose(self, X, alg="svd", rank=0, opt_trunc=False, tikhonov=0, sorting="abs", Y=None, dt=None, observables=None):

        if observables is not None:
            for i in range(observables):
                if i == 0:
                    X = observables[0](X)
                    Y = observables[0](Y)
                X = np.vtstack((X, observables[i](X)))
                Y = np.vtstack((X, observables[i](Y)))

        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)

        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]

        A1 = Y @ X.T
        A2 = X @ X.T
        # POD Decomposition of the (X X.T) matrix
        self.pod_ = POD()
        self.pod_.decompose(A2, alg=alg, rank=rank,
                            opt_trunc=opt_trunc)
        u = self.pod_.modes
        vh = self.pod_.time
        s = self.pod_.singvals
        s_inv = np.zeros(s.shape)
        s_inv = 1 / s
        s_inv_ = s_inv.copy()
        if self.tikhonov:
            s_inv_ *= s**2 / (s**2 + self.tikhonov * self.x_cond)
        A2_pinv = np.linalg.multi_dot((vh, s_inv_, u.T))
        self._kept_rank = self.pod_.kept_rank

        # Computing the Koopman operator approximation
        self.__A = A1 @ A2_pinv

        # Eigendecomposition on the Koopman operator
        lambd, w = np.linalg.eig(self.A)
        if sorting == "abs":
            idx = (np.abs(lambd)).argsort()[::-1]
        else:
            idx = (np.real(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        # Computing the high-dimensional DMD modes [1]
        phi = w.copy()
        omega = np.log(lambd) / dt  # Continuous system eigenvalues

        # Loading the DMD instance's attributes
        self.dt = dt
        self.dmd_modes = phi
        self.lambd = lambd
        self.eigenvalues = omega
        self.singvals = s
        self.modes = u
        self.time = vh

        return u, s, vh

    def _compute_amplitudes(self, t1, method):
        self.t1 = t1
        init = self.init
        b, _, _, _ = np.linalg.lstsq(self.dmd_modes, init, rcond=None)
        b /= np.exp(self.eigenvalues * t1)
        return b
