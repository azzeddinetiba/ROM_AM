import imp
import numpy as np
from scipy import interpolate
from .pod import POD
from .dmd import DMD
from .hodmd import HODMD
from copy import deepcopy
import warnings


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
        """

        Parameters
        ----------
        X : numpy.ndarray
            Parametric snapshot matrix data, of (p, N, m) size
        params : numpy.ndarray
            Parameters in a (p, ) array

        """

        self._p = X.shape[0]  # Number of parameters samples
        self._N = X.shape[1]  # Number of dofs
        self._m = X.shape[2]  # Number of timesteps

        self.stacked_X = X.swapaxes(0, 2).swapaxes(
            0, 1).reshape((self._N, self._m*self._p), order='F')  # of size (N, m * p)

        self.params = params

        # POD Decomposition of the stacked X's POD coefficients
        self.pod_ = POD()
        self.pod_.decompose(self.stacked_X, alg=alg, rank=rank1,
                            opt_trunc=opt_trunc)
        u = self.pod_.modes
        vh = self.pod_.time
        s = self.pod_.singvals
        self._kept_rank = self.pod_.kept_rank

        self.pod_coeff = np.diag(s) @ vh

        self.stacked_coeff = self.pod_coeff.swapaxes(0, 1).reshape(
            (self._p, self._m, self._kept_rank),).swapaxes(1, 2).reshape((-1, self._m))  # of size (n * p, m)

        if not partitioned:
            if dmd_model == "dmd":
                # DMD Decomposition
                self.dmd_model = DMD()
                _, _, _ = self.dmd_model.decompose(self.stacked_coeff[:, :-1], Y=self.stacked_coeff[:, 1::],
                                                   alg=alg, dt=dt, rank=rank2, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)

            elif dmd_model == "hodmd":
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
                    tmp_model = DMD()
                    _, _, _ = tmp_model.decompose(self.stacked_coeff[i*self._kept_rank:(i+1)*self._kept_rank, :-1], Y=self.stacked_coeff[i*self._kept_rank:(i+1)*self._kept_rank, 1::],
                                                  alg=alg, dt=dt, rank=rank2, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)
                elif dmd_model == "hodmd":
                    tmp_model = HODMD()
                    _, _, _ = tmp_model.decompose(self.stacked_coeff[i*self._kept_rank:(i+1)*self._kept_rank, :-1], Y=self.stacked_coeff[i*self._kept_rank:(i+1)*self._kept_rank, 1::],
                                                  alg=alg, dt=dt, rank=rank2, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting, hod=hod)

                self.dmd_model.append(deepcopy(tmp_model))
                self.A_tilde.append(tmp_model.A)

        return u, s, vh

    def predict(self, t, mu, t1, rank=None, stabilize=False):

        if not self.is_Partitioned:
            sample_res = self.dmd_model.predict(
                t=t, t1=t1, method=0, rank=rank, stabilize=stabilize)  # of shape (n * p, m)

        else:
            sample_res = np.empty((self._kept_rank * self._p, t.shape[0]))
            for i in range(self._p):
                sample_res[i*self._kept_rank:(i+1)*self._kept_rank, :] = self.dmd_model[i].predict(
                    t=t, t1=t1, method=0, rank=rank, stabilize=stabilize)

        f = interpolate.interp1d(self.params, sample_res.reshape(
            (self._p, self._kept_rank, -1)).T.swapaxes(0, 1), kind='cubic')  # sample_res shaped towards (n, m, p)

        return self.pod_.modes @ f(mu)
