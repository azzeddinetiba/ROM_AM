import numpy as np
import scipy.linalg as sp
import sys

os_ = 1
if 'linux' in sys.platform:
    os_ = 0
    import jax.numpy as jnp
    import jax

    jax.config.update('jax_platform_name', 'cpu')


class ROM:

    def __init__(self, rom="pod"):

        self.rom = rom

    def decompose(self, X, center=False, alg="svd"):

        if center:
            X -= X.mean(axis=0)

        if alg == "svd":
            if os_ == 0:
                u, s, vh = jnp.linalg.svd(X, False)
            else:
                u, s, vh = sp.svd(X, False)
        elif alg == "cov":

            cov = X.T @ X
            lambd, v = np.linalg.eigh(cov)
            lambd = np.flip(lambd)
            v = np.flip(v, 1)
            lambd[abs(lambd) < 6e-11] = 0

            s = np.sqrt(lambd)
            vh = v.T
            s_inv = np.zeros(s.shape)

            # --> Compute the pseudo-inverse.
            s_inv[abs(s) > 6e-11] = 1.0 / s[abs(s) > 6e-11]

            u = X @ v @ np.diag(s_inv)

        self.singvals = s
        self.modes = u
        self.time = vh

    def approximate(self, rank=0):

        if rank == 0:
            rank = len(self.singvals)

        return np.linalg.multi_dot((self.modes[:, :rank], np.diag(self.singvals[:rank]), self.time[:rank, :]))
