import numpy as np
from .dmd import DMD
from scipy.spatial import KDTree
from scipy.optimize import nnls


class CnvxDMD:

    def __init__(self):
        pass

    def decompose(self,
                  X,
                  params,
                  dt=None,):
        """Training the Convex hull dynamic mode decomposition model 
        using the input data X and the training parameters params

        Parameters
        ----------
        X : numpy.ndarray
            Parametric snapshot matrix data, of (p, N, m) size
        params : numpy.ndarray
            Parameters in a (k, p) array
        dt : float
            value of time step from each column in X to the next column
        """

        self._p = X.shape[0]  # Number of parameters samples
        self._N = X.shape[1]  # Number of dofs
        self._m = X.shape[2]  # Number of timesteps
        self._k = params.shape[0]

        self.data = X
        self.params = params
        self.dt = dt

        self.param_tree = KDTree(params.T)

        u = 0
        vh = 0
        s = 0

        return u, s, vh

    def predict(self,
                t,
                mu,
                t1,
                rank_pred=None,
                alg="svd",
                rank=None,
                opt_trunc=False,
                tikhonov=0,
                sorting="abs",
                nnls_tikhonov=None,
                stabilize=False,
                init=None,
                k=None,
                method=0,
                cutoff=1.):
        """Predict the CnvxDMD solution on the prescribed time instants and 
        the target aprameter value.

        Parameters
        ----------
        t : numpy.ndarray, size (nt, )
            time steps at which the parDMD solution will be computed
        mu : numpy.darray, size(k, 1)
            Parameter value for prediction
        t1: float
            the value of the time instant of the first data snapshot
            If 'method=1' is used and t1 indicates the time isntant 
            when the solution corresponds to 'init'
            Default 0.
        rank : None, int or float, optional
            if rank = 0 or rank is None All the ranks are kept, 
            unless their singular values are zero
            if 0 < rank < 1, it is used as the percentage of
            the energy that should be kept, and the rank is
            computed accordingly
            Default : None
        alg : str, optional
            Whether to use the SVD on decomposition ("svd") or
            the eigenvalue problem on snaphot matrices ("snap")
            Default : "svd"
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
        stabilize : bool, optional
            DMD eigenvalue-shifting to stable eigenvalues at the prediction
            phase
            Default : True
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
        rank_pred: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None
        nnls_tikhonov: None or float
            tikhonov parameter for regularization
            If None, no regularization is applied, if float, it is used as
            the lambda tikhonov parameter for the weights finding problem
            Default : None
        k: None or int
            Number of neighbors used for reconstruction of the parameters in
            the parameter space
            if None k = dim + 1 where dim is the dimension of the parameter space
            Default : None

        Returns
        ----------
            numpy.ndarray, size (N, nt)
            parDMD solution on the time values t and parameter value mu
        """

        if k is None:
            k = self._k + 1

        _, ii = self.param_tree.query(mu.ravel(), k=k)
        new_snaps = self.data[ii, :, :]

        weights = self.cnvx_nnls(mu, self.params[:, ii], mu=nnls_tikhonov)

        weighted_snaps = np.dot(new_snaps.T, weights).T

        # DMD Decomposition
        dmd_model = DMD()
        u, s, vh = dmd_model.decompose(weighted_snaps[:, :-1], Y=weighted_snaps[:, 1::],
                                       alg=alg, dt=self.dt, rank=rank, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)

        return dmd_model.predict(t, t1=t1, method=method, rank=rank_pred, stabilize=stabilize, init=init, cutoff=cutoff)

    def cnvx_nnls(self, z, Z, ksi=1e5, mu=None):

        k = Z.shape[1]
        ksi_ = ksi * np.trace(Z.T @ Z)/k

        Zaug = np.vstack((Z, np.sqrt(ksi_) * np.ones((1, k))))
        zaug = np.vstack((z, np.array([[np.sqrt(ksi_)]])))

        if mu is not None:
            mu_ = mu * np.trace(Zaug.T @ Zaug)/k
            Zaug = np.vstack((Zaug, mu * np.eye(k+1)))
            zaug = np.vstack((zaug, np.zeros((k+1, 1))))

        w, _ = nnls(Zaug, zaug.ravel())

        return w
