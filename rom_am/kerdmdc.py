import numpy as np
import warnings


class KERDMDC:
    """
    Kernel DMD with actuation

    """

    def kerfun(self, X, Y, kernel=None, p=2, a=1, sig=1):
        if kernel is None:
            kernel = self.kernel
        if kernel == "poly":
            XTY = (1 + X.T @ Y)**p
            XTX = (1 + X.T @ X)**p
        elif kernel == "sigmoid":
            XTY = np.tanh(X.T @ Y + a)
            XTX = np.tanh(X.T @ X + a)
        elif kernel == "gaussian":
            XTY = np.exp(-X.T @ Y)
            XTX = np.exp(-X.T @ X)
        elif kernel == "radial":
            XTY = np.exp(-(X-Y).T @ (X-Y) / sig**2)
            XTX = XTY

        return XTY, XTX

    def __init__(self):

        self.singvals = None
        self.modes = None
        self.time = None
        self.dmd_modes = None
        self.dt = None
        self.t1 = None
        self.n_timesteps = None
        self.tikhonov = None
        self.x_cond = None
        self._kept_rank = None
        self.init = None
        self.A_tilde = None
        self._A = None
        self._no_reduction = False
        self.pred_rank = None
        self._pod_coeff = None
        self.stock = False
        self.data = None

    def decompose(self, X,
                  alg="snap",
                  rank=0,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  Y=None,
                  dt=None,
                  Y_input=None,
                  kernel="poly",
                  p=2,
                  a=1,
                  sig=1,
                  inputfun=None):
        """
        Y_input : numpy.ndarray
            Control inputs matrix data, of (, m) size
            organized as 'm' snapshots

        """
        self.n_timesteps = X.shape[1]
        self.data_X = X
        self.init = X[:, 0]
        self.inputfun = inputfun
        self.sorting = sorting
        self.kernel = kernel

        # Computing X.T @ Y/X using chosen kernels
        XTY, XTX = self.kerfun(X, Y, kernel, p, a, sig)

        xi_u_inv = np.diag(inputfun(1/(Y_input.ravel())))

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
        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)
        if self.tikhonov:
            s_inv *= s**2 / (s**2 + self.tikhonov * self.x_cond)

        v = v[:, :rank]
        vh = v.T

        # Compute K^
        self._A_chap = np.linalg.multi_dot(
            (np.diag(s_inv), vh, XTY, xi_u_inv, vh.T, np.diag(s_inv)))

        # Loading the DMD instance's attributes
        self.dt = dt
        self.singvals = s
        self.inv_singv = s_inv
        self.time = vh
        self.modes = None

        return 0, s, vh

    def A(self, u):
        """Computes the high dimensional DMD operator.

        """
        return self._A_chap * self.inputfun(u)

    def koopman(self, u):

        A = self.A(u)

        lambd, w = np.linalg.eig(A)
        _, ksi = np.linalg.eig(A.T)

        sorting = self.sorting
        if sorting == "abs":
            idx = (np.abs(lambd)).argsort()[::-1]
        else:
            idx = (np.real(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        ksi = ksi[:, idx]
        low_dim_eig = w

        modes = np.linalg.multi_dot(
            (self.data_X, self.time.T, np.diag(self.inv_singv), low_dim_eig))
        eigfuncoef = self.time.T @ np.diag(self.inv_singv) @ ksi

        return lambd, modes, eigfuncoef

    def predict(self, u_input):
        """Predict the DMD solution on the prescribed time instants.

        """

        t_size = u_input.shape[1]
        pred = np.empty((self.data_X.shape[0], t_size+1), dtype=complex)
        pred[:, 0] = self.init

        for i in range(t_size):
            lambd, modes, eigfuncoef = self.koopman(u_input[:, i])
            xty, _ = self.kerfun(pred[:, i].reshape((-1, 1)),
                                 self.data_X)
            b = xty @ eigfuncoef
            pred[:, i+1] = (modes @ np.diag(lambd) @ b.T).ravel()

        return pred
