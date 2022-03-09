import numpy as np
from .dmd import DMD
from .pod import POD
import warnings


class HODMD(DMD):
    """
    High Order Dynamic Mode Decomposition Class

    """

    def decompose(self, X, alg="svd", rank=0, opt_trunc=False, tikhonov=0, sorting="abs", Y=None, dt=None, observables=None, hod=50):

        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)

        # POD Decomposition of the X matrix
        self.pod_ = POD()
        self.pod_.decompose(X, alg=alg, rank=rank,
                            opt_trunc=opt_trunc)
        u = self.pod_.modes
        vh = self.pod_.time
        s = self.pod_.singvals
        self._ho_kept_rank = self.pod_.kept_rank

        new_X = u.T @ np.hstack((X, Y[:, -1].reshape((-1, 1))))
        ho_X_ = np.zeros((hod * new_X.shape[0], new_X.shape[1]+1-hod))

        for i in range(hod):
            ho_X_[i*X.shape[0]:(i+1) * X.shape[0],
                  :] = new_X[:X.shape[0], i:i+(new_X.shape[1]+1-hod)]

        ho_X = ho_X_[:, :-1]
        ho_Y = ho_X_[:, 1::]
        _, _, _ = super().decompose(ho_X,
                                    alg=alg,
                                    rank=0,
                                    opt_trunc=opt_trunc,
                                    tikhonov=0,
                                    sorting=sorting,
                                    Y=ho_Y,
                                    dt=dt,)

        # Loading the HODMD instance's attributes, overriding DMD
        self.singvals = s
        self.modes = u
        self.time = vh
        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]
        self.dmd_modes = u @ self.low_dim_eig[:self._ho_kept_rank, :]

        return u, s, vh
