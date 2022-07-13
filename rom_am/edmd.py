import numpy as np
from .dmd import DMD
from .pod import POD
import warnings


class EDMD(DMD):
    """
    Extended Dynamic Mode Decomposition Class

    """

    def __init__(self):
        super().__init__()
        self._accuracy = None
        self.tall = False

    def decompose(self, X, alg="svd", rank=None, opt_trunc=False, tikhonov=0, sorting="abs", Y=None, dt=None, observables=None):
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
        rank : None, int or float, optional
            if rank = 0 All the ranks are kept, unless their
            singular values are zero
            if 0 < rank < 1, it is used as the percentage of
            the energy that should be kept, and the rank is
            computed accordingly
            Default : None
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

        """
        if observables is not None:
            if "X" in observables:
                for i in range(len(observables["X"])):
                    if i == 0:
                        tempX = observables["X"][0](X)
                    else:
                        tempX = np.vstack((tempX, observables["X"][i](X)))
                X = tempX
            if "Y" in observables:
                for i in range(len(observables["Y"])):
                    if i == 0:
                        tempY = observables["Y"][0](Y)
                    else:
                        tempY = np.vstack((tempY, observables["Y"][i](Y)))
                Y = tempY

        self._rectangular = False
        if X.shape != Y.shape:
            self._train_data = X
            self._trained_on = Y
            self._rectangular = True
        self._dim_Y = Y.shape[0]
        self._dim_X = X.shape[0]

        if X.shape[1] <= X.shape[0]:
            self.tall = True

        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)

        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]

        if self.tall and not self._rectangular:
            u, s, vh = super().decompose(X=X, alg=alg, rank=rank, opt_trunc=opt_trunc,
                                         tikhonov=tikhonov, sorting=sorting, Y=Y, dt=dt)
        else:
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
            if not self._rectangular:
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
                self.dmd_modes = phi
                self.lambd = lambd
                self.eigenvalues = omega

            # Loading the DMD instance's attributes
            self.dt = dt
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

    def predict(self, t, t1=0, method=1, rank=None, stabilize=True, x_input=None, init=None):
        """Predict the eDMD solution on the prescribed time instants.

        Parameters
        ----------
        t : numpy.ndarray, size (nt, )
            time steps at which the DMD solution will be computed
            If x_input is given, this argument is disregarded
        t1: float
            the value of the time instant of the first data snapshot
            If 'method=1' is used and t1 indicates the time isntant 
            when the solution corresponds to 'init'
            If x_input is given, this argument is disregarded
            Default 0.
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None
            If x_input is given, this argument is disregarded
        method: int
            Method used to compute the initial mode amplitudes
            0 if it is computed on the POD subspace as in Tu et al.[1]
            1 if it is computed using the pseudoinverse of the DMD modes
            Default : 0
            If x_input is given, this argument is disregarded
        stabilize : bool, optional
            DMD eigenvalue-shifting to stable eigenvalues at the prediction
            phase
            Default : False
            If x_input is given, this argument is disregarded
        u_input: numpy.ndarray, size (n, nt)
            Values of X data in the prediction phase. Should be given
            in case the eDMD operator is rectangular, i.e X and Y do not have
            the same dimensions
        init : int or ndarray or None, optional
            The initial condition used to compute the amplitudes
            it is an :
                - int when used with 'method = 0', representing the index
                of the snapshot used from the snapshots training data.
                Note that here t1 will be taken as the time insant at that
                snapshot (init * dt), so the t1 argument is here the time
                instant of the first snapshot data.
                - ndarray of size (n, ) when used with 'method = 1',
                representing the prescribed initial condition at t = t1 (It has
                to be prescribed accordingly).
                - None, then the first data snapshot will be used
                whether in 'method = 0' or 'method = 1'
            Default None
        Returns
        ----------
            numpy.ndarray, size (N, nt)
            ROM solution on the time values t
        """
        if self._rectangular:
            return self.A @ x_input
        else:
            return super().predict(t, t1, method, rank, stabilize, init)

    def reconstruct(self, rank=None):
        if self._rectangular:
            return self.predict(t=0, x_input=self._train_data)
        else:
            return super().reconstruct(rank)

    @property
    def accuracy(self,):
        return self.get_accuracy()

    def get_accuracy(self, rank=None):
        if self._accuracy is None:
            err = np.linalg.norm(self.reconstruct(
                rank=rank) - self._trained_on, axis=0)/np.linalg.norm(self._trained_on, axis=0)
            self._accuracy = err.sum()/err.shape[0]
        return self._accuracy
