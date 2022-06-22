import numpy as np
from scipy.interpolate import RBFInterpolator
from .pod import POD
from .dmd import DMD
from .hodmd import HODMD
from copy import deepcopy


class ManDMD:

    def __init__(self) -> None:
        self.is_Partitioned = False

    def log_A(self, U, Ut):

        N = U.shape[0]

        prod = POD()
        u, _, rh = prod.decompose(Ut.T @ U)
        procr = np.linalg.multi_dot((Ut, u, rh))
        L = (np.eye(N) - U @ U.T) @ (procr)

        #M = U.T @ Ut
        #L = Ut @ np.linalg.inv(M) - U
        prod2 = POD()
        q, sig, vh = prod2.decompose(L)
        
        sig_norm = 1.1 * np.abs(sig).max()

        return sig_norm * np.linalg.multi_dot((q, np.diag(np.arcsin(sig/sig_norm)), vh))
        #return np.linalg.multi_dot((q, np.diag(np.arctan(sig)), vh))

    def exp_A(self, U, delt):

        prod = POD()
        q, sig, vh = prod.decompose(delt[0, :, :])

        return np.linalg.multi_dot((U, vh.T, np.diag(np.cos(sig)), vh)) + np.linalg.multi_dot((q, np.diag(np.sin(sig)), vh))

    def decompose(self,
                  X,
                  params,
                  alg="svd",
                  rank=0,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  dt=None,
                  dmd_model="dmd",
                  hod=50,
                  kernel='thin_plate_spline',
                  epsilon=None):
        """Training the dynamic mode decomposition with manifold 
        interpolation using the input data X and the training 
        parameters params

        Parameters
        ----------
        X : numpy.ndarray
            Parametric snapshot matrix data, of (p, N, m) size
        params : numpy.ndarray
            Parameters in a (k, p) array
        dt : float
            value of time step from each column in X to the next column
        alg : str, optional
            Whether to use the SVD on decomposition ("svd") or
            the eigenvalue problem on snaphot matrices ("snap")
            Default : "svd"
        rank1 : int or float, optional
            if rank1 = 0 All the ranks are kept, unless their
            singular values are zero
            if 0 < rank < 1, it is used as the percentage of
            the energy that should be kept, and the rank is
            computed accordingly
            Default : 0
        rank2 : int or float, optional
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
        """

        self._p = X.shape[0]  # Number of parameters samples
        self._N = X.shape[1]  # Number of dofs
        self._m = X.shape[2]  # Number of timesteps

        self.params = params
        self.dt = dt

        for i in range(self._p):

            # DMD Decomposition
            dmd_model = DMD()
            u, s, vh = dmd_model.decompose(X[i, :, :-1], Y=X[i, :, 1::],
                                           alg=alg, dt=dt, rank=rank, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)

            if i == 0:

                stacked_Ulog = np.empty(
                    (self._p, self._N, dmd_model._kept_rank))
                stacked_Alog = np.empty(
                    (self._p, dmd_model._kept_rank, dmd_model._kept_rank))

                rank = dmd_model._kept_rank

                ref_U = dmd_model.modes
                ref_A = dmd_model.A_tilde

                stacked_Ulog[i, :, :] = dmd_model.modes
                stacked_Alog[i, :, :] = dmd_model.A_tilde

            else:

                stacked_Ulog[i, :, :] = self.log_A(ref_U, dmd_model.modes)
                stacked_Alog[i, :, :] = self.log_A(ref_A, dmd_model.A_tilde)

        self.f_U = RBFInterpolator(
            self.params.T, stacked_Ulog, kernel=kernel, epsilon=epsilon)
        self.f_A = RBFInterpolator(
            self.params.T, stacked_Alog, kernel=kernel, epsilon=epsilon)
        self._kept_rank = rank
        self.ref_U = ref_U
        self.ref_A = ref_A

        return u, s, vh

    def predict(self, t, mu, t1, rank=None, stabilize=False, method=0, init=None):
        """Predict the parDMD solution on the prescribed time instants and 
        the target aprameter value.

        Parameters
        ----------
        t : numpy.ndarray, size (nt, )
            time steps at which the parDMD solution will be computed
        mu : numpy.darray, size(k, 1)
            Parameter value for prediction
        t1: float
            the value of the time instant of the first snapshot
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None 

        Returns
        ----------
            numpy.ndarray, size (N, nt)
            parDMD solution on the time values t and parameter value mu
        """

        U_pred = self.exp_A(self.ref_U, self.f_U(mu))
        A_pred = self.exp_A(self.ref_A, self.f_A(mu))

        dmd_pred = DMD()
        self.load_DMD(dmd_pred, U_pred, A_pred, init)

        self.dpd = dmd_pred
        if method == 2:
            raise NotImplementedError
        if method == 1:
            if init is None:
                raise Exception(
                    "An initial condition 'init' should be prescribed in case of 'method=1' for prediction")

        return dmd_pred.predict(t, t1, method, rank, stabilize, init)

    def load_DMD(self, dmd_model, modes, A_tilde, initial):

        dmd_model.modes = modes
        dmd_model.A_tilde = A_tilde
        dmd_model._kept_rank = self._kept_rank

        # Eigendecomposition on the low dimensional operator
        lambd, w = np.linalg.eig(A_tilde)
        idx = (np.abs(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        dmd_model.low_dim_eig = w
        # Computing the high-dimensional DMD modes [1]
        dmd_model.dmd_modes = modes @ w
        # Continuous system eigenvalues
        dmd_model.eigenvalues = np.log(lambd) / self.dt
        dmd_model.lambd = lambd

        dmd_model.init = initial
