import numpy as np
from .pod import POD
import warnings


class DMD:
    """
    Dynamic Mode Decomposition Class

    """

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

    def decompose(self,
                  X,
                  alg="svd",
                  rank=0,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  Y=None,
                  dt=None,
                  no_reduc=False):
        """Training the dynamic mode decomposition[1] model, using the input data X and Y

        Parameters
        ----------
        X : numpy.ndarray
            Snapshot matrix data, of (N, m) size
        Y : numpy.ndarray
            Second Snapshot matrix data, of (N, m) size
            advanced from X by dt
        dt : float
            value of time step from each snapshot in X
            to each snapshot in Y
        alg : str, optional
            Whether to use the SVD on decomposition ("svd") or
            the eigenvalue problem on snaphot matrices ("snap")
            Default : "svd"
        rank : int or float, optional
            if rank = 0 All the ranks are kept, unless their
            singular values are zero
            if 0 < rank < 1, it is used as the percentage of
            the energy that should be kept, and the rank is
            computed accordingly
            Default : 0
        opt_trunc : bool, optional
            if True an optimal truncation/threshold is estimated,
            based on the algorithm of Gavish and Donoho [2]
            Default : False
        tikhonov : int or float, optional
            tikhonov parameter for regularization
            If 0, no regularization is applied, if float, it is used as
            the lambda tikhonov parameter
            Default : 0
        sorting : str, optional
            Whether to sort the discrete DMD eigenvalues by absolute
            value ("abs") or by their real part ("real")
            Default : "abs"

        References
        ----------

        [1] On dynamic mode decomposition:  Theory and applications,
        Journal of Computational Dynamics,1,2,391,421,2014-12-1,
        Jonathan H. Tu,Clarence W. Rowley,Dirk M. Luchtenburg,
        Steven L. Brunton,J. Nathan Kutz,2158-2491_2014_2_391,

        [2] M. Gavish and D. L. Donoho, "The Optimal Hard Threshold for
        Singular Values is 4/sqrt(3) ," in IEEE Transactions on Information
        Theory, vol. 60, no. 8, pp. 5040-5053, Aug. 2014,
        doi: 10.1109/TIT.2014.2323359.

        Returns
        ------
        u : numpy.ndarray, of size(N, r)
            The spatial modes of the training data

        s : numpy.ndarray, of size(r, )
            The singular values of the training data

        vh : numpy.ndarray, of size(r, m)
            The time dynamics of the training data


        """
        if X.shape[0] != Y.shape[0]:
            raise Exception("The DMD input snapshots X and Y have to have \
                the same observables, for different observables in X and Y, \
                    Please consider using the EDMD() class")
        if X.shape[1] != Y.shape[1]:
            raise Exception("The DMD input snapshots X and Y have to have \
                the same number of instants")

        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)

        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]

        if not no_reduc:
            # POD Decomposition of the X matrix
            self.pod_ = POD()
            self.pod_.decompose(X, alg=alg, rank=rank,
                                opt_trunc=opt_trunc)
            u = self.pod_.modes
            vh = self.pod_.time
            s = self.pod_.singvals
            self._kept_rank = self.pod_.kept_rank

            # Computing the A_tilde: the projection of the 'A' operator
            # on the POD modes, where A = Y * pseudoinverse(X) [1]
            s_inv = np.zeros(s.shape)
            s_inv = 1 / s
            if self.tikhonov:
                s_inv *= s**2 / (s**2 + self.tikhonov * self.x_cond)
            store = np.linalg.multi_dot((Y, vh.T, np.diag(s_inv)))
            self.A_tilde = u.T @ store

            # Eigendecomposition on the low dimensional operator
            lambd, w = np.linalg.eig(self.A_tilde)
            if sorting == "abs":
                idx = (np.abs(lambd)).argsort()[::-1]
            else:
                idx = (np.real(lambd)).argsort()[::-1]
            lambd = lambd[idx]
            w = w[:, idx]
            self.low_dim_eig = w

            # Computing the high-dimensional DMD modes [1]
            phi = store @ w
        else:  # Should ONLY be used in the context of parametric DMD
            self._no_reduction = True
            self._A = Y @ np.linalg.pinv(X)
            lambd, phi = np.linalg.eig(self._A)

            s = 0.
            u = 0.
            vh = 0.

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

    def predict(self, t, t1=0, method=0, rank=None, stabilize=True):
        """Predict the DMD solution on the prescribed time instants.

        Parameters
        ----------
        t : numpy.ndarray, size (nt, )
            time steps at which the DMD solution will be computed
        t1: float
            the value of the time instant of the first snapshot
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None
        method: int
            Method used to compute the initial mode amplitudes
            0 if it is computed on the POD subspace as in Tu et al.[1]
            1 if it is computed using the pseudoinverse of the DMD modes
            Default : 0
        stabilize : bool, optional
            DMD eigenvalue-shifting to stable eigenvalues at the prediction
            phase
            Default : True

        Returns
        ----------
            numpy.ndarray, size (N, nt)
            DMD solution on the time values t
        """
        if self.dmd_modes is None:
            raise Exception("The DMD decomposition hasn't been executed yet")

        if rank is None:
            rank = self._kept_rank
        elif not (isinstance(rank, int) and 0 < rank < self.kept_rank):
            warnings.warn('The rank chosen for prediction should be an integer smaller than the\
            rank chosen/computed at the decomposition phase. Please see the rank value in self.kept_rank')
            rank = self._kept_rank

        b = self._compute_amplitudes(t1, method)

        eig = self.eigenvalues[:rank]
        if stabilize:
            eig_rmpl = eig[np.abs(self.lambd[:rank]) > 1]
            eig_rmpl.real = 0
            eig[np.abs(self.lambd[:rank]) > 1] = eig_rmpl

        return self.dmd_modes[:, :rank] @ (np.exp(np.outer(eig, t).T) * b[:rank]).T

    def reconstruct(self, rank=None):
        """Reconstruct the data input using the DMD Model.

        Parameters
        ----------
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None 

        Returns
        ----------
            numpy.ndarray, size (N, m)
            DMD solution on the time steps where the input snapshots are taken
        """
        if rank is None:
            rank = self._kept_rank
        elif not (isinstance(rank, int) and 0 < rank < self.kept_rank):
            warnings.warn('The rank chosen for reconstruction should be an integer smaller than the\
            rank chosen/computed at the decomposition phase. Please see the rank value in self.kept_rank')
            rank = self._kept_rank

        if self.t1 is None:
            self.t1 = 0
            warnings.warn('the initial instant value was not assigned during the prediction phase,\
                t1 is chosen as 0')

        t = np.linspace(self.t1 + self.dt, self.t1 + (self.n_timesteps - 1)
                        * self.dt, self.n_timesteps)
        return self.predict(t, t1=self.t1)

    def _compute_amplitudes(self, t1, method):
        self.t1 = t1
        if method == 1:
            init = self.init
            b, _, _, _ = np.linalg.lstsq(self.dmd_modes, init, rcond=None)
            b /= np.exp(self.eigenvalues * t1)
        elif method == 2:
            L = self.low_dim_eig[:self._ho_kept_rank, :] @ np.tile(np.eye(self.lambd.shape[0]), self.n_timesteps) * np.tile(self.lambd, self.n_timesteps)**np.repeat(
                np.linspace(1, self.n_timesteps, self.n_timesteps, dtype=int), self.lambd.shape[0])
            L = np.vstack((self.low_dim_eig[:self._ho_kept_rank, :], L.reshape(
                self._ho_kept_rank, -1, self.lambd.shape[0]).swapaxes(0, 1).reshape((-1, self.lambd.shape[0]))))
            b, _, _, _ = np.linalg.lstsq(
                L, (self.modes.T @ self.data).reshape((-1, 1), order='F').ravel(), rcond=None)
            b /= np.exp(self.eigenvalues * t1)
        else:
            alpha1 = self.singvals * self.time[:, 0]
            b = np.linalg.solve(self.lambd * self.low_dim_eig, alpha1) / np.exp(
                self.eigenvalues * t1
            )
        return b

    @property
    def A(self):
        """Computes the high dimensional DMD operator.

        """
        if self._A is None:
            self._A = self.modes @ self.A_tilde @ self.modes.T
        return self._A
