import numpy as np
from rom_am.pod import POD

def _compute_past(Om, L, z, beta):

    h = np.linalg.multi_dot([Om.T, z])
    K = np.linalg.multi_dot([h.T, L]) / \
        (beta + np.linalg.multi_dot([h.T, L, h]))
    K = np.reshape(K, (1, -1))
    h = np.reshape(h, (-1, 1))

    L = (L - np.linalg.multi_dot([L, h, K])) / beta
    Om = Om + np.linalg.multi_dot([(z - np.linalg.multi_dot([Om, h])), K])

    return Om, L

class RPOD(POD):

    def __init__(self, lambdaForget=0.99):
        super().__init__()
        self.lambdaForget = lambdaForget
        self.L = None

    def decompose(self, X, alg="svd", rank=None, opt_trunc=False, tikhonov=0, thin=False):

        u, s, vh = super().decompose(X, alg, rank, opt_trunc, tikhonov, thin)
        self.L = self.pod_coeff @ self.pod_coeff.T
        return u, s, vh

    def update(self, newX):

        self.modes, self.L = _compute_past(
            self.modes, self.L, newX, self.lambdaForget)
