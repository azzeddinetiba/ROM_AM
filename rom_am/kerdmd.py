import numpy as np
from numpy import s_
from rom_am.dmd import DMD


class KERDMD(DMD):

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

    def decompose(self, X,
                  alg="snap",
                  rank=None,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  Y=None,
                  dt=None,
                  kernel="poly",
                  p=2,
                  a=1,
                  sig=1,
                  kerfun=None):
        """Training the Kernel Order dynamic mode decomposition[1] model,
        using the input data X and Y

        Parameters
        ----------
        kernel : string
            The kernel used the ones implemented are :
            "poly" Polynomial kernels with parameters 'a' and 'p' : f(x, y) = (a+x'y)^p
            "sigmoid" kernel with 'a' parameter : f(x, y) = tanh(x'y + a)
            "gaussian" kernels : f(x, y) = exp(-x'y)
            "radial" Radial basis kernels with 'sig' paramter : f(x, y) = exp(-|x-y|²/sig²)
        p : int
            Kernel parameter
            Default : 2
        a : int
            Kernel parameter
            Default : 1
        sig : int
            Kernel parameter
            Default : 1
        kerfun : lambda function
            User-specific kernel function defined as f(x, y) that nd.arrays and where
            f(x, y) is equivalent to X'Y
            Default : None

        References
        ----------

        [1] A kernel-based method for data-driven koopman spectral analysis,
        Journal of Computational Dynamics,2,2,247,265,2016-5-1,
        Matthew O.  Williams,Clarence W. Rowley,Ioannis G.  Kevrekidis,

        """
        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]
        self.X_data = X
        self.sorting = sorting

        # Defining the used kernels
        self.defker(kerfun, kernel=kernel, p=p, a=a, sig=sig)

        # Computing X.T @ Y/X using chosen kernels
        XTY = self.kernel(X, Y)
        XTX = self.kernel(X, X)

        # Eigendecomposition of X.T @ Y to compute singular values and time coefficients
        vh, s, s_inv = self.snap_svd(XTX, rank, opt_trunc, tikhonov)

        # Computing the Koopman operator approximation
        self._A = np.linalg.multi_dot(
            (np.diag(s_inv), vh, XTY, vh.T, np.diag(s_inv)))
        # The A here is equivalent to the A_tilde defined in DMD
        # as the nonlinear terms (and the according dimension)
        # are not formulated explicitly
        self.A_tilde = self._A

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
        self.singvals_inv = s_inv
        self.time = vh
        self.dmd_modes = phi
        self.lambd = lambd
        self.eigenvalues = omega
        self.modes = None

        return 0, s, vh

    def snap_svd(self, mat, rank, opt_trunc, tikhonov):
        vals, v = np.linalg.eigh(mat)

        vals = np.flip(vals)
        v = np.flip(v, 1)
        vals[vals < 1e-10] = 0
        s = np.sqrt(vals)

        if opt_trunc:
            if self.init.shape[0] <= self.n_timesteps:
                beta = self.init.shape[0]/self.n_timesteps
            else:
                beta = self.n_timesteps/self.init.shape[0]
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

        self._kept_rank = rank

        return vh, s, s_inv

    @property
    def left_eigvectors(self):
        """Returns the left eigenvectors of the DMD operator.

        """
        print("The left eigenvectors are not available in KerDMD, \
            because the observables are not known explicitly")
        raise NotImplementedError

    def koop_eigf(self, x):
        """Computes the Koopman eigenfunction at x

        Parameters
        ----------

        x: ndarray, of shape (N, nt)
            m points of N dimension at which eigenfunctions will
            be computes
            N must be the same as the dimension of snapshots

        Returns
        ----------
            numpy.ndarray, size (k, nt)
            the k eigenfunctions computed at the x points

        """

        return np.linalg.multi_dot((self.low_dim_left_eig.T,
                                    np.diag(self.singvals_inv), self.time,
                                    self.kernel(x, self.X_data).T))
