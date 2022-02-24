import numpy as np
from .dmd import DMD
from .pod import POD
import warnings


class EDMD(DMD):
    """
    Extended Dynamic Mode Decomposition Class

    """

    def decompose(self, X, alg="svd", rank=0, opt_trunc=False, tikhonov=0, sorting="abs", Y=None, dt=None, observables=None):
        """Training the extended dynamic mode decomposition[1, 3] model, using the input data X and Y
            Cases where the number of timesteps is much bigger than the number of observables
            considered.

        Parameters
        ----------
        X : numpy.ndarray
            Snapshot matrix data, of (N, m) size,
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
        observables : list, optional
            List of Lambda functions defining the observable expression
            from the state variables
            if None, the eDMD considers the input snapshots
            Default : None

        References
        ----------

        [1] Williams, M.O., Kevrekidis, I.G. & Rowley, C.W.
        A Data-Driven Approximation of the Koopman Operator: Extending
        Dynamic Mode Decomposition. J Nonlinear Sci 25, 1307-1346 (2015).
        doi : 10.1007/s00332-015-9258-5
        https://doi.org/10.1007/s00332-015-9258-5

        [2] M. Gavish and D. L. Donoho, "The Optimal Hard Threshold for
        Singular Values is 4/sqrt(3) ," in IEEE Transactions on Information
        Theory, vol. 60, no. 8, pp. 5040-5053, Aug. 2014,
        doi: 10.1109/TIT.2014.2323359.

        [3] Kutz, J. & Brunton, Steven & Brunton, Bingni & Proctor, Joshua.
        (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. 
        https://epubs.siam.org/doi/book/10.1137/1.9781611974508

        Returns
        ------
        u : numpy.ndarray, of size(N, r)
            POD modes of X * X.T

        s : numpy.ndarray, of size(r, )
            The singular values of X * X.T

        vh : numpy.ndarray, of size(r, m)
            The time dynamics of X * X.T

        Notes
        -----
        It is advised to use this class only when the number of timesteps is much
        bigger than the number of observables considered.

        """
        if X.shape[1] <= X.shape[0]:
            warnings.warn("The input snapshots are tall and skinny, consider DMD for this kind of problems.\
                 eDMD is best suited for fat and short matrices")

        if observables is not None:
            for i in range(len(observables)):
                if i == 0:
                    tempX = observables[0](X)
                    tempY = observables[0](Y)
                else:
                    tempX = np.vstack((tempX, observables[i](X)))
                    tempY = np.vstack((tempY, observables[i](Y)))
            X = tempX
            Y = tempY

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
        A2_pinv = np.linalg.multi_dot((vh.T, np.diag(s_inv_), u.T))
        self._kept_rank = self.pod_.kept_rank

        # Computing the Koopman operator approximation
        self._A = A1 @ A2_pinv

        # Eigendecomposition on the Koopman operator
        lambd, w = np.linalg.eig(self.A)
        if sorting == "abs":
            idx = (np.abs(lambd)).argsort()[::-1]
        else:
            idx = (np.real(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        # Computing the high-dimensional DMD modes
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
