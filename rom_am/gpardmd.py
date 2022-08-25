from hashlib import new
import imp
from turtle import st
import numpy as np
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
from .pod import POD
from .dmd import DMD
from .hodmd import HODMD
from copy import deepcopy


class GParDMD:

    def __init__(self) -> None:
        self.is_Partitioned = False

    def decompose(self,
                  X,
                  params,
                  dt,
                  alg="svd",
                  rank=None,
                  opt_trunc=False,
                  rankpar=None,
                  kernel='thin_plate_spline',
                  epsilon=None,):
        """Training the Parametric dynamic mode decomposition model 
        using the input data X and the training parameters params

        Parameters
        ----------

        """
        self.params = params
        self.dt = dt

        self._p = X.shape[0]  # Number of parameters samples
        self._N = X.shape[1]  # Number of dofs
        self._m = X.shape[2]  # Number of timesteps

        # Parametric POD Decomposition of the stacked X
        self.par_stacked_X = X.T.reshape((-1, self._p))
        self.pod_ = POD()
        self.pod_.decompose(self.par_stacked_X, alg=alg, rank=rankpar,
                            opt_trunc=opt_trunc)
        self.param_modes = self.pod_.time.T

        # Spacial POD Decomposition of the stacked X
        self.spc_stacked_X = X.swapaxes(1, 2).reshape((-1, self._N)).T
        self.pod_ = POD()
        self.pod_.decompose(self.spc_stacked_X, alg=alg, rank=rank,
                            opt_trunc=opt_trunc)
        self.spc_modes = self.pod_.modes

        self.pod_coeff = np.einsum(
            'ab,cak,cd->bdk', self.spc_modes, X, self.param_modes)

        self.f_phi = RBFInterpolator(
            self.params.T, self.param_modes, kernel=kernel, epsilon=epsilon)

        return self.spc_modes, self.pod_.singvals, self.pod_.time

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
                stabilize=False,
                init=None,
                method=0,
                cutoff=1.):
        """Predict the parDMD solution on the prescribed time instants and 
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

        new_snaps = np.einsum('ab,bck,c->ak', self.spc_modes,
                              self.pod_coeff, self.f_phi(mu.T).ravel())

        dmd_model = DMD()
        _, _, _ = dmd_model.decompose(new_snaps[:, :-1], Y=new_snaps[:, 1::],
                                      alg=alg, dt=self.dt, rank=rank, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)

        return dmd_model.predict(t=t, t1=t1, method=method, rank=rank_pred, stabilize=stabilize, init=init, cutoff=cutoff)
