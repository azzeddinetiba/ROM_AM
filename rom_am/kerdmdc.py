import numpy as np
from .pod import POD
import warnings


class KERDMDC:
    """
    Kernel DMD with actuation

    """

    def defker(self, kerfun=None, kernel="poly", p=2, a=1, sig=1):
        if kernel is None:
            self.kernel = kerfun
        if kernel == "poly":
            self.kernel = lambda x, y: (1 + x.T @ y)**p
        elif kernel == "sigmoid":
            self.kernel = lambda x, y: np.tanh(x.T @ y + a)
        elif kernel == "gaussian":
            self.kernel = lambda x, y: np.exp(-x.T @ y)
        elif kernel == "radial":
            self.kernel = lambda x, y: np.exp((np.einsum(
                'ij,ij->j', x, x)[:, None] + np.einsum('ij,ij->j', y, y)[None, :] - 2 * np.dot(x.T, y))/(-sig**2))

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
        self._koop_modes = None
        self._koop_eigf_coeff = None
        self._koop_state_coeff = None

    def decompose(self, X,
                  alg="snap",
                  rank=None,
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
                  kerfun=None):
        """
        Y_input : numpy.ndarray
            Control inputs matrix data, of (, m) size
            organized as 'm' snapshots

        """
        self.n_timesteps = X.shape[1]
        self.nx = X.shape[0]
        self.data_X = X
        self.init = X[:, 0]
        self.sorting = sorting
        self.kernel = kernel

        Xin = np.vstack((X, Y_input))  # X_input including the control inputs
        Yout = Y  # Y_out has the vector state only
        self.Yout = Yout
        self.Xin = Xin

        # Defining the used kernels
        self.defker(kerfun, kernel=kernel, p=p, a=a, sig=sig)

        # Computing X.T @ Y/X using chosen kernels
        XiTXi = self.kernel(Xin, Xin)
        XoTXo = self.kernel(Yout, Yout)

        # Eigendecomposition of X.T @ Y to compute singular values and time coefficients
        vhi, si, s_invi = self.snap_svd(XiTXi, rank, opt_trunc, tikhonov)
        vho, so, s_invo = self.snap_svd(XoTXo, rank, opt_trunc, tikhonov)

        # Compute K^
        self._A = np.linalg.multi_dot(
            (np.diag(so), vho, vhi.T, np.diag(s_invi)))
        # The A here is equivalent to the A_tilde defined in DMD
        # as the nonlinear terms (and the according dimension)
        # are not formulated explicitly

        # SVD of the A_tilde matrix
        svdA = POD()
        q_tilde, sig_, v_tilde = svdA.decompose(
            self._A, rank=rank, tikhonov=tikhonov)
        self._kept_rank = svdA.kept_rank

        # Loading the DMD instance's attributes
        self.lambd = sig_
        self.dt = dt
        self.singvals = so
        self.inv_singv = s_invo
        self.s_invi = s_invi
        self.vi = vhi.T
        self.time = vho
        self.modes = q_tilde
        self.rgt_modes = v_tilde.T

        _ = self.koop_state_coeff

        return q_tilde, so, vho

    @property
    def koop_eigv(self):
        """Returns the koopman eigenvalues.

        """
        if self._koop_eigv is None:
            self._koop_eigv = self.lambd
        return self._koop_eigv

    @property
    def koop_modes(self):
        """Returns the koopman modes.

        """
        if self._koop_modes is None:
            self._koop_modes = np.linalg.multi_dot(
                (self.Yout, self.time.T, np.diag(self.inv_singv), self.modes))
        return self._koop_modes

    @property
    def koop_eigf_coeff(self):
        """Returns the Koopman modes.

        """
        if self._koop_eigf_coeff is None:
            self._koop_eigf_coeff = np.linalg.multi_dot(
                (self.vi, np.diag(self.s_invi), self.rgt_modes))
        return self._koop_eigf_coeff

    def koop_eigf(self, x):
        """Computes the Koopman eigenfunction at x

        """
        f = self.kernel(x, self.Xin)
        return self.koop_eigf_coeff.T @ f.T

    @property
    def koop_state_coeff(self):
        """Returns the koopman eigenvalues.

        """
        if self._koop_state_coeff is None:
            self._koop_state_coeff = np.linalg.multi_dot(
                (self.koop_modes[:self.nx, :], np.diag(self.koop_eigv), self.koop_eigf_coeff.T))
        return self._koop_state_coeff

    def predict(self, t, t1=0, rank=None, u_input=None):
        """Predict the DMD solution on the prescribed time instants.

        """

        t_size = u_input.shape[1]
        pred = np.empty((self.nx, t_size+1), dtype=complex)
        pred[:, 0] = self.init
        for i in range(t_size):
            xin = np.vstack((pred[:, i].reshape((-1, 1)),
                            u_input[:, i].reshape((-1, 1))))
            pred[:, i+1] = (self.koop_state_coeff @
                            self.kernel(xin, self.Xin).T).ravel()

        return pred

    def snap_svd(self, mat, rank, opt_trunc, tikhonov):
        vals, v = np.linalg.eigh(mat)

        vals = np.flip(vals)
        v = np.flip(v, 1)
        vals[vals < 1e-10] = 0
        s = np.sqrt(vals)

        if opt_trunc:
            if self.nx <= self.n_timesteps:
                beta = self.nx/self.n_timesteps
            else:
                beta = self.n_timesteps/self.nx
            omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
            tau = np.median(s) * omega
            rank = np.sum(s > tau)
        else:
            if rank is None:
                rank = len(vals[vals > 1e-10])
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
            self.x_cond = np.linalg.cond(mat)
        if self.tikhonov:
            s_inv *= s**2 / (s**2 + self.tikhonov * self.x_cond)

        v = v[:, :rank]
        vh = v.T

        return vh, s, s_inv
