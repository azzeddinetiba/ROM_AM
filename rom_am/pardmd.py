import imp
import numpy as np
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
from .pod import POD
from .dmd import DMD
from .hodmd import HODMD
from copy import deepcopy


class ParDMD:

    def __init__(self) -> None:
        self.is_Partitioned = False

    def decompose(self,
                  X,
                  params,
                  alg="svd",
                  rank1=0,
                  rank2=0,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  dt=None,
                  dmd_model="dmd",
                  hod=50,
                  partitioned=True):
        """Training the Parametric dynamic mode decomposition[model, 
        using the input data X and the training parameters params

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

        self.stacked_X = X.swapaxes(0, 2).swapaxes(
            0, 1).reshape((self._N, self._m*self._p), order='F')  # of size (N, m * p)
        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(self.stacked_X)

        self.params = params
        self._k = params.shape[1]

        # POD Decomposition of the stacked X's POD coefficients
        self.pod_ = POD()
        self.pod_.decompose(self.stacked_X, alg=alg, rank=rank1,
                            opt_trunc=opt_trunc)
        u = self.pod_.modes
        vh = self.pod_.time
        s = self.pod_.singvals
        s_ = s.copy()
        if self.tikhonov:
            s_ = (s**2 + self.tikhonov * self.x_cond) / s
        self._kept_rank = self.pod_.kept_rank

        self.pod_coeff = np.diag(s_) @ vh

        self.stacked_coeff = self.pod_coeff.swapaxes(0, 1).reshape(
            (self._p, self._m, self._kept_rank),).swapaxes(1, 2).reshape((-1, self._m))  # of size (n * p, m)

        if not partitioned:
            if dmd_model == "dmd":
                # DMD Decomposition
                self.dmd_model = DMD()
                _, _, _ = self.dmd_model.decompose(self.stacked_coeff[:, :-1], Y=self.stacked_coeff[:, 1::],
                                                   alg=alg, dt=dt, rank=rank2, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)

            elif dmd_model == "hodmd":
                # High-Order DMD Decomposition
                self.dmd_model = HODMD()
                _, _, _ = self.dmd_model.decompose(self.stacked_coeff[:, :-1], Y=self.stacked_coeff[:, 1::],
                                                   alg=alg, dt=dt, rank=rank2, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting, hod=hod)

            self.A_tilde = self.dmd_model.A
        else:

            self.is_Partitioned = True
            self.A_tilde = []
            self.dmd_model = []
            for i in range(self._p):

                if dmd_model == "dmd":
                    # DMD Decomposition
                    tmp_model = DMD()
                    _, _, _ = tmp_model.decompose(self.stacked_coeff[i*self._kept_rank:(i+1)*self._kept_rank, :-1], Y=self.stacked_coeff[i*self._kept_rank:(i+1)*self._kept_rank, 1::],
                                                  alg=alg, dt=dt, rank=rank2, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)
                elif dmd_model == "hodmd":
                    # High-Order DMD Decomposition
                    tmp_model = HODMD()
                    _, _, _ = tmp_model.decompose(self.stacked_coeff[i*self._kept_rank:(i+1)*self._kept_rank, :-1], Y=self.stacked_coeff[i*self._kept_rank:(i+1)*self._kept_rank, 1::],
                                                  alg=alg, dt=dt, rank=rank2, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting, hod=hod)

                self.dmd_model.append(deepcopy(tmp_model))
                self.A_tilde.append(tmp_model.A)

        return u, s, vh

    def predict(self, t, mu, t1, rank=None, stabilize=False, kernel='thin_plate_spline', method=0):
        """Predict the parDMD solution on the prescribed time instants and 
        the target aprameter value.

        Parameters
        ----------
        t : numpy.ndarray, size (nt, )
            time steps at which the parDMD solution will be computed
        mu : float
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
        if not self.is_Partitioned:
            sample_res = self.dmd_model.predict(
                t=t, t1=t1, method=0, rank=rank, stabilize=stabilize)  # of shape (n * p, m)

        else:
            sample_res = np.empty(
                (self._kept_rank * self._p, t.shape[0]), dtype=complex)
            for i in range(self._p):
                sample_res[i*self._kept_rank:(i+1)*self._kept_rank, :] = self.dmd_model[i].predict(
                    t=t, t1=t1, method=method, rank=rank, stabilize=stabilize)

        # f = interpolate.interp1d(self.params, sample_res.reshape(
        #      (self._p, self._kept_rank, -1)).T.swapaxes(0, 1), kind='cubic')  # sample_res shaped towards (n, m, p)

        f = RBFInterpolator(self.params, sample_res.reshape(
            (self._p, self._kept_rank, -1)).T.swapaxes(0, 1).T, kernel=kernel)
        self.tst = sample_res.reshape(
            (self._p, self._kept_rank, -1)).T.swapaxes(0, 1).T
        return self.pod_.modes @ f(mu).T[:, :, 0]
