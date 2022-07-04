from weakref import ref
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.linalg import logm, expm
from .pod import POD
from .dmd import DMD
from .hodmd import HODMD
from copy import deepcopy


class ManDMD:

    def __init__(self) -> None:
        self.exp = False
        self.is_Partitioned = False

    def log_A(self, A, At):
        if self.exp:
            return logm(At @ np.linalg.inv(A))
        else:
            return At - A

    def exp_A(self, A, At):
        if self.exp:
            return expm(At) @ A
        else:
            return A + At

    def log_U(self, U, Ut):

        N = U.shape[0]

        prod = POD()
        u, _, rh = prod.decompose(Ut.T @ U,)
        procr = np.linalg.multi_dot((Ut, u, rh))
        L = (np.eye(N) - U @ U.T) @ (procr)

        prod2 = POD()
        q, sig, vh = prod2.decompose(L, thin=True)

        return np.linalg.multi_dot((q, np.diag(np.arcsin(sig)), vh))

    def exp_U(self, U, delt):

        prod = POD()
        q, sig, vh = prod.decompose(delt, thin=True)

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
                  epsilon=None,
                  iref=0,
                  method=2,
                  exp=False):
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

        if exp:
            self.exp = True
        # DMD Decomposition
        dmd_model = DMD()
        u, s, vh = dmd_model.decompose(X[iref, :, :-1], Y=X[iref, :, 1::],
                                       alg=alg, dt=dt, rank=rank, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)
        if method == 2 or method == 4:
            b = dmd_model._compute_amplitudes(method=2)

        stacked_Ulog = np.empty(
            (self._p, self._N, dmd_model._kept_rank))
        stacked_Alog = np.empty(
            (self._p, dmd_model._kept_rank, dmd_model._kept_rank))
        if method == 2 or method == 4:
            stacked_b = np.empty(
                (self._p, dmd_model._kept_rank, ), dtype=complex)
        if method == 4:
            stacked_phi = np.empty(
                (self._p, self._N, dmd_model._kept_rank), dtype=complex)

        rank = dmd_model._kept_rank

        ref_U = dmd_model.modes
        ref_A = dmd_model.A_tilde

        stacked_Ulog[iref, :, :] = np.zeros((self._N, rank))
        stacked_Alog[iref, :, :] = np.zeros((rank, rank))

        if method == 2 or method == 4:
            stacked_b[iref, :] = b
        if method == 4:
            stacked_phi[iref, :, :] = dmd_model.dmd_modes

        for i in np.delete(np.arange(0, self._p), iref):

            # DMD Decomposition
            dmd_model = DMD()
            u, s, vh = dmd_model.decompose(X[i, :, :-1], Y=X[i, :, 1::],
                                           alg=alg, dt=dt, rank=rank, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)
            b = dmd_model._compute_amplitudes(method=2)

            stacked_Ulog[i, :, :] = self.log_U(ref_U, dmd_model.modes)
            stacked_Alog[i, :, :] = self.log_A(ref_A, dmd_model.A_tilde)
            if method == 2 or method == 4:
                stacked_b[i, :] = b
            if method == 4:
                stacked_phi[i, :, :] = dmd_model.dmd_modes

        self.f_U = RBFInterpolator(
            self.params.T, stacked_Ulog, kernel=kernel, epsilon=epsilon)
        self.f_A = RBFInterpolator(
            self.params.T, stacked_Alog, kernel=kernel, epsilon=epsilon)
        if method == 2 or method == 4:
            self.f_b = RBFInterpolator(
                self.params.T, stacked_b, kernel=kernel, epsilon=epsilon)
        if method == 4:
            self.f_phi = RBFInterpolator(
                self.params.T, stacked_phi, kernel=kernel, epsilon=epsilon)
        self._kept_rank = rank
        self.ref_U = ref_U
        self.ref_A = ref_A

        return u, s, vh

    def predict(self, t, mu, t1, rank=None, stabilize=False, method=0, init=None, cutoff=1.):
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

        given_b = False
        b = None
        phi = None
        U_pred = self.exp_U(self.ref_U, self.f_U(mu.T)[0, :, :])

        A_pred = self.exp_A(self.ref_A, self.f_A(mu.T)[0, :, :])
        if method == 2 or method == 4:
            b = self.f_b(mu.T)[0, :]
        if method == 4 or method == 14:
            phi = self.f_phi(mu.T)[0, :, :]
        dmd_pred = DMD()
        self.load_DMD(dmd_pred, U_pred, A_pred, init, b, phi)

        self.dpd = dmd_pred
        if method == 0:
            raise NotImplementedError
        if method == 1:
            if init is None:
                raise Exception(
                    "An initial condition 'init' should be prescribed in case of 'method=1' for prediction")
        if method == 2 or method == 4:
            given_b = True
        if method == 14:
            method = 1

        return dmd_pred.predict(t, t1, method, rank, stabilize, init, given_b=given_b, cutoff=cutoff)

    def load_DMD(self, dmd_model, modes, A_tilde, initial, b, phi=None):

        dmd_model.modes = modes
        dmd_model.A_tilde = A_tilde
        dmd_model._kept_rank = self._kept_rank
        dmd_model.computed_amplitudes = b

        # Eigendecomposition on the low dimensional operator
        lambd, w = np.linalg.eig(A_tilde)
        idx = (np.abs(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        dmd_model.low_dim_eig = w
        # Computing the high-dimensional DMD modes [1]
        if phi is not None:
            dmd_model.dmd_modes = phi
        else:
            dmd_model.dmd_modes = modes @ w
        # Continuous system eigenvalues
        dmd_model.eigenvalues = np.log(lambd) / self.dt
        dmd_model.lambd = lambd

        dmd_model.init = initial
