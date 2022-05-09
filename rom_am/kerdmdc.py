import numpy as np
from .pod import POD
import warnings


class KERDMDC:
    """
    Kernel DMD with actuation

    """

    def kerfun(self, X, Y, kernel=None, p=2, a=1, sig=1):
        if kernel is None:
            kernel = self.kernel
        if kernel == "poly":
            self.kernel = lambda x, y: (1 + x.T @ y)**p
            XTY = self.kernel(X, Y)
            XTX = self.kernel(X, X)
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
        self._koop_eigv = None
        self._koop_mode = None
        self._koop_eifg_coeff = None

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
        self.nx = X.shape[0]
        self.data_X = X
        self.init = X[:, 0]
        self.inputfun = inputfun
        self.sorting = sorting
        self.kernel = kernel

        Xin = np.vstack((X, Y_input))
        Yout = Y.copy()
        self.Yout = Yout
        self.Xin = Xin

        # Computing X.T @ Y/X using chosen kernels
        _, XoTXo = self.kerfun(Yout, Yout, kernel, p, a, sig)
        _, XiTXi = self.kerfun(Xin, Xin, kernel, p, a, sig)

        # Eigendecomposition of X.T @ Y to compute singular values and time coefficients
        valso, vo = np.linalg.eigh(XoTXo)
        valso = np.flip(valso)
        vo = np.flip(vo, 1)
        valso[valso < 1e-10] = 0
        so = np.sqrt(valso)

        if rank == 0:
            rank = len(valso[valso > 1e-10])
        elif 0 < rank < 1:
            rank = np.searchsorted(
                np.cumsum(so**2 / (so**2).sum()), rank) + 1

        so = so[:rank]
        s_invo = np.zeros(so.shape)
        s_invo[so > 1e-10] = 1.0 / so[so > 1e-10]
        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)
        if self.tikhonov:
            s_invo *= so**2 / (so**2 + self.tikhonov * self.x_cond)

        vo = vo[:, :rank]
        vho = vo.T

        # Eigendecomposition of X.T @ Y to compute singular values and time coefficients
        valsi, vi = np.linalg.eigh(XiTXi)

        valsi = np.flip(valsi)
        vi = np.flip(vi, 1)
        valsi[valsi < 1e-10] = 0
        si = np.sqrt(valsi)

        if rank == 0:
            rank = len(valsi[valsi > 1e-10])
        elif 0 < rank < 1:
            rank = np.searchsorted(
                np.cumsum(si**2 / (si**2).sum()), rank) + 1

        si = si[:rank]
        s_invi = np.zeros(si.shape)
        s_invi[si > 1e-10] = 1.0 / si[si > 1e-10]
        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)
        if self.tikhonov:
            s_invi *= si**2 / (si**2 + self.tikhonov * self.x_cond)

        vi = vi[:, :rank]
        vhi = vi.T

        # Compute K^
        self._A = np.linalg.multi_dot((np.diag(so), vho, vi, np.diag(s_invi)))

        svdA = POD()
        q_tilde, sig_, v_tilde = svdA.decompose(self._A, rank=rank, tikhonov = tikhonov)

        # Loading the DMD instance's attributes
        self.lambd = sig_
        self.dt = dt
        self.singvals = so
        self.inv_singv = s_invo
        self.s_invi = s_invi
        self.vi = vi
        self.time = vho
        self.modes = q_tilde
        self.rgt_modes = v_tilde.T

        return q_tilde, so, vho

    @property
    def koop_eigv(self):
        """Returns the koopman eigenvalues.

        """
        if self._koop_eigv is None:
            self._koop_eigv = self.lambd
        return self._koop_eigv

    @property
    def koop_mode(self):
        """Returns the koopman modes.

        """
        if self._koop_mode is None:
            self._koop_mode = np.linalg.multi_dot(
                (self.Yout, self.time.T, np.diag(self.inv_singv), self.modes))
        return self._koop_mode

    @property
    def koop_eifg_coeff(self):
        """Returns the koopman modes.

        """
        if self._koop_eifg_coeff is None:
            self._koop_eifg_coeff = np.linalg.multi_dot(
                (self.vi, np.diag(self.s_invi), self.rgt_modes))
        return self._koop_eifg_coeff

    def koop_eigf(self, x):
        f = self.kernel(x, self.Xin)
        return (f @ self.koop_eifg_coeff).T

    def predict(self, t, t1=0, rank=None, u_input=None):
        """Predict the DMD solution on the prescribed time instants.

        """

        t_size = u_input.shape[1]
        pred = np.empty((self.nx, t_size+1), dtype=complex)
        pred[:, 0] = self.init
        store = self.koop_mode[:self.nx, :] @ np.diag(self.koop_eigv)
        for i in range(t_size):
            xin = np.vstack((pred[:, i].reshape((-1, 1)),
                            u_input[:, i].reshape((-1, 1))))
            pred[:, i+1] = (store @ self.koop_eigf(xin)).ravel()

        return pred
