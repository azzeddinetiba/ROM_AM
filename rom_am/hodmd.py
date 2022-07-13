import numpy as np
from .dmd import DMD
from .pod import POD
import warnings


class HODMD(DMD):
    """
    High Order Dynamic Mode Decomposition Class

    """

    def decompose(self,
                  X,
                  alg="svd",
                  rank=None,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  Y=None,
                  dt=None,
                  hod=50):
        """Training the High Order dynamic mode decomposition[1] model,
        using the input data X and Y

        Parameters
        ----------
        hod : int
            number of previous snapshots to take into account in the
            DMD decomposition
            Default : 50

        References
        ----------

        [1] S. Le Clainche and J. M. Vega. Higher order dynamic mode
        decomposition. SIAM Journal on Applied Dynamical Systems,
        16(2):882-925, 2017

        """
        if hod <= 0 or (not isinstance(hod, int) and not isinstance(hod, np.int64)):
            raise ValueError("Invalid 'hod' value, it should be an integer greater "
                             "than 0")
        if hod > X.shape[1]:
            raise ValueError("Invalid 'hod' value, it cannot be greater than the number "
                             "of snapshots")
        r = rank
        if rank is None:
            r = min(X.shape[0], X.shape[1])
        if hod <= X.shape[1] and ((X.shape[1]+1-hod) < 0.01 * (r*hod)):
            warnings.warn("The 'd' (hod) value is too close to the number of snapshots, "
                          "it does not enable the modified snapshot matrix to have "
                          "enough snapshots, the results may be erroneous ")

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

        self.data = np.hstack((X, Y[:, -1].reshape((-1, 1))))
        # ===== POD coefficients of the training data; size (r, Nt+1) =====
        s_ = s.copy()
        if self.tikhonov:
            s_ = (s**2 + self.tikhonov * self.x_cond) / s
        new_X = np.diag(s_) @ vh

        # ===== Augmented matrix to contain Hankel bloc matrices of POD coefficients; size (dr, Nt + 1 - d)=====
        ho_X_ = np.empty((hod * new_X.shape[0], new_X.shape[1]+1-hod))

        for i in range(hod):
            ho_X_[i*self._ho_kept_rank:(i+1) * self._ho_kept_rank,
                  :] = new_X[:, i:i+(new_X.shape[1]+1-hod)]

        ho_X = ho_X_[:, :-1]
        ho_Y = ho_X_[:, 1::]
        _, _, _ = super().decompose(ho_X,
                                    alg=alg,
                                    rank=None,
                                    opt_trunc=opt_trunc,
                                    tikhonov=0,
                                    sorting=sorting,
                                    Y=ho_Y,
                                    dt=dt,)  # No truncation for the second reduction for now

        # Loading the HODMD instance's attributes, overriding DMD's
        self.ho_modes = self.modes.copy()
        self.singvals = s
        self.modes = u
        self.time = vh
        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]
        self.phi = self.dmd_modes
        self.dmd_modes = u[:, :min(
            self._ho_kept_rank, self._kept_rank)] @ self.low_dim_eig[:self._ho_kept_rank, :]

        return u, s, vh

    def predict(self, t, t1=0, rank=None, stabilize=False, method=2):
        self.pred_rank = min(self._ho_kept_rank, self._kept_rank)
        # ==== Only method = 2 for now ====
        if method != 2:
            method = 2
            warnings.warn(
                "The method 2 is the only one suppported for HoDMD, It will be used here.")
        return super().predict(t=t, t1=t1, method=method, rank=rank, stabilize=stabilize)

    @property
    def A(self):
        """Computes the high dimensional DMD operator.

        """
        if self._A is None:
            self._A = self.ho_modes @ self.A_tilde @ self.ho_modes.T
        return self._A
