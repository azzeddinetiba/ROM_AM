import numpy as np
import scipy.linalg as sp
import sys

os_ = 1
if "linux" in sys.platform:
    os_ = 0
    import jax.numpy as jnp
    import jax

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)


class POD:
    def decompose(self, X, center=False, alg="svd", rank=0, opt_trunc=False, tikhonov=0):

        if center:
            self.mean_flow = X.mean(axis=1)
            X -= self.mean_flow.reshape((-1, 1))

        if alg == "svd":
            if os_ == 0:
                u, s, vh = jnp.linalg.svd(X, False)
                u = np.array(u)
                s = np.array(s)
                vh = np.array(vh)
            else:
                u, s, vh = sp.svd(X, False)

            if opt_trunc:
                if X.shape[0] <= X.shape[1]:
                    beta = X.shape[0]
                else:
                    beta = X.shape[1]
                omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
                tau = np.median(s) * omega
                rank = np.sum(s > tau)
            else:
                if rank == 0:
                    rank = len(s[s > 1e-10])
                elif 0 < rank < 1:
                    rank = np.searchsorted(
                        np.cumsum(s**2 / (s**2).sum()), rank) + 1

            u = u[:, :rank]
            vh = vh[:rank, :]
            s = s[:rank]

        elif alg == "snap":
            cov = X.T @ X
            lambd, v = np.linalg.eigh(cov)

            lambd = np.flip(lambd)
            v = np.flip(v, 1)
            lambd[lambd < 1e-10] = 0

            if rank == 0:
                rank = len(lambd[lambd > 1e-10])

            v = v[:, :rank]
            vh = v.T

            s = np.sqrt(lambd[:rank])
            s_inv = np.zeros(s.shape)
            s_inv[s > 1e-10] = 1.0 / s[s > 1e-10]

            u = X @ v[:, :rank] * s_inv

        self.singvals = s
        self.modes = u
        self.time = vh

        self.tikhonov = tikhonov

        return u, s, vh

    def reconstruct(self, rank=0):

        if rank == 0:
            rank = len(self.singvals)

        return (self.modes[:, :rank] * self.singvals[:rank]) @ self.time[:rank, :]
