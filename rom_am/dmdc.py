import numpy as np
from .pod import POD


class DMDc:
    """
    Dynamic Mode Decomposition with Control Class

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
        """Training the dynamic mode decomposition with control model,
                    using the input data X and Y, and Y_input

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
        center : bool, optional
            Flag to either center the data around time or not
            Default : False
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
        Y_input : numpy.ndarray
            Control inputs matrix data, of (q, m) size
            organized as 'm' snapshots

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
            The singular values modes of the training data

        vh : numpy.ndarray, of size(r, m)
            The time dynamics of the training data


        """
        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)

        # POD Decomposition of the X and Y matrix
        Omega = np.vstack((X, Y_input))
        self.pod_til = POD()
        self.pod_hat = POD()
        u_til, s_til, vh_til = self.pod_til.decompose(
            Omega, center = center, alg=alg, rank=rank, opt_trunc=opt_trunc)
        u_til_1 = u_til[: X.shape[0], :]
        u_til_2 = u_til[: Y_input.shape[0], :]
        u_hat, _, _ = self.pod_hat.decompose(
            Y, center = center, alg=alg, rank=rank, opt_trunc=opt_trunc)

        s_til_inv = np.zeros(s_til.shape)
        s_til_inv = 1 / s_til
        s_til_inv_ = s_til_inv.copy()
        if self.tikhonov:
            s_til_inv_ *= s_til**2 / (s_til**2 + self.tikhonov * self.x_cond)
        store_ = np.linalg.multi_dot((Y, vh_til.T, np.diag(s_til_inv_)))
        store = u_hat.T @ store_
        self.A_tilde = np.linalg.multi_dot((store, u_til_1.T, u_hat))
        self.B_tilde = store @ u_til_2.T

        # Eigendecomposition on the low dimensional operators
        lambd, w = np.linalg.eig(self.A_tilde)
        if sorting == "abs":
            idx = (np.abs(lambd)).argsort()[::-1]
        else:
            idx = (np.real(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        # Computing the exact DMDc modes
        phi = np.linalg.multi_dot((store_, u_til_1.T, u_hat, w))
        omega = np.log(lambd) / dt

        # Loading the DMDc instance's attributes
        u = u_til_1
        vh = vh_til
        s = s_til
        self.dmd_modes = phi
        self.lambd = lambd
        self.eigenvalues = omega
        self.singvals = s
        self.modes = u
        self.time = vh
        self.u_hat = u_hat

        return u, s, vh

    def predict(self, t, t1=0, rank=None, x_input=None, u_input=None):
        """Predict the DMD solution on the prescribed time instants.

        Parameters
        ----------
        t: numpy.ndarray, size (nt, )
            time steps at which the DMD solution will be computed
        t1: float
            the value of the time instant of the first snapshot
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None
        x_input: numpy.ndarray, size (N, nt)
            state matrix at time steps t
        x_input: numpy.ndarray, size (q, nt)
            control input matrix at time steps t

        Returns
        ----------
            numpy.ndarray, size (N, nt)
            DMDc solution on the time values t+dt
        """

        return self.u_hat @ (self.A_tilde @ self.u_hat.T @ x_input
                             + self.B_tilde @ u_input)
