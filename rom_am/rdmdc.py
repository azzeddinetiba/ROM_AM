import numpy as np
from rom_am.rpod import RPOD
from rom_am.rom import ROM


class RDMDC:

    def __init__(self, lambdaForget=0.99, epsilon=0.01, lambdaForgetBasis=0.99):
        self.lambdaForget = lambdaForget
        self.epsilon = epsilon
        self.lamnbdaForgetBasis = lambdaForgetBasis

    def decompose(self,
                  X,
                  Y,
                  alg               = "svd",
                  rank              = None,
                  opt_trunc         = False,
                  tikhonov          = 0,
                  u_input           = None,
                  initialization    = 0,
                  normLoadReducer   = False,
                  precomp_mean      = None,
                  precomp_std       = None,
                  precomputed_modes = None):

        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)

        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]
        self.input_init = u_input[:, 0]

        # POD Decomposition of the X and Y matrix
        self.pod = RPOD(self.lamnbdaForgetBasis, epsilon = self.epsilon)
        self.rom = ROM(self.pod)
        if precomp_mean is not None:
            self.rom.decompose(
                Y, alg=alg, rank=rank, opt_trunc=opt_trunc,
                precomp_mean=precomp_mean, precomp_std=precomp_std)
        else:
            self.rom.decompose(
                Y, alg=alg, rank=rank, opt_trunc=opt_trunc,
                normalize=normLoadReducer, center=True)
        if precomputed_modes is not None:
            self.pod.modes = precomputed_modes
        self._kept_rank = self.pod.kept_rank

        Xr = self.pod.project(self.rom.normalize(self.rom.center(X)))
        Yr = self.pod.project(self.rom.normalize(self.rom.center(Y)))

        input_ = np.vstack((Xr, u_input))

        # Normalization   ---------------------
        input_max = np.max(np.abs(input_), axis=1)[
            :, np.newaxis]
        input_ = input_ / input_max
        self.input_max = input_max
        # -------------------------------------

        self.abTilde = np.linalg.lstsq(input_.T, Yr.T)[0].T
        newsize = self.pod.kept_rank*(self.pod.kept_rank+u_input.shape[0])

        if not initialization:
            self.L = self.epsilon * np.eye(newsize)
        else:
            self.L = np.zeros((newsize, newsize))
            for i in range(u_input.shape[1]):
                uTilde = np.vstack((self.pod.pod_coeff[:, [i]], u_input[:, [i]]))
                phi = uTilde.reshape((1, -1))
                phi = np.tile(phi, (self.pod.kept_rank, self.pod.kept_rank)).T
                self.L += phi @ phi.T
            self.L = np.linalg.inv(self.L)

    def predict(self, previousX, u_input):

        input_1 = self.pod.project(
            self.rom.normalize(self.rom.center(previousX)))
        input_2 = u_input
        input_ = np.vstack((input_1, input_2))
        input_ = input_/self.input_max

        return self.rom.decenter(self.rom.denormalize(self.pod.inverse_project(self.abTilde @ input_)))

    def update(self, newX, previousX, new_input):

        uTilde = np.vstack(
            (self.pod.project(self.rom.normalize(self.rom.center(previousX))), new_input))
        uTilde = uTilde/self.input_max
        self.pod.update(self.rom.normalize(self.rom.center(newX)))

        phi = uTilde.reshape((1, -1))
        phi = np.tile(phi, (self.pod.kept_rank, self.pod.kept_rank)).T

        K = np.linalg.multi_dot([self.L, phi, np.linalg.inv(
            self.lambdaForget * np.eye(self.pod.kept_rank) + np.linalg.multi_dot([phi.T, self.L, phi]))])
        self.L = self.L - \
            np.linalg.multi_dot([K, phi.T, self.L])/self.lambdaForget

        self.abTilde = self.matricize(self.vectorize(
            self.abTilde) + K @ (self.pod.project(self.rom.normalize(self.rom.center(newX))) - phi.T @ self.vectorize(self.abTilde)))

    def vectorize(self, x):
        return x.reshape((-1, 1))

    def matricize(self, x):
        return x.reshape((self.pod.kept_rank, -1))
