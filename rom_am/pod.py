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

    def decompose(self, X, Y=None, dt=None, center=False, alg="svd", rank=0):

        if center:
            X -= X.mean(axis=0)

        if self.rom == "pod":
            u, s, vh = self._pod_decompose(X, alg, rank)

        elif self.rom == "dmd":
            u, s, vh, lambd, phi = self._dmd_decompose(X, Y, dt, rank)
            omega = np.log(lambd)/dt

            self.dmd_modes = phi
            self.eigenvalues = omega

        self.singvals = s
        self.modes = u
        self.time = vh

    def _dmd_decompose(X, Y, dt, rank=0):

        if os_ == 0:
            u, s, vh = jnp.linalg.svd(X, False)
        else:
            u, s, vh = sp.svd(X, False)

        s_inv = np.zeros(s.shape)
        s_inv[s > 1e-10] = 1 / s[s > 1e-10]
        A_tilde = np.linalg.multi_dot((u.T, Y, vh.T, s_inv))

        lambd, w = np.linalg.eig(A_tilde)

        phi = np.linalg.multi_dot((Y, vh.T, s_inv, w))

        return u, s, vh, lambd, phi

    def _pod_decompose(self, X, alg, rank=0):

        if alg == "svd":
            if os_ == 0:
                u, s, vh = jnp.linalg.svd(X, False)
            else:
                u, s, vh = sp.svd(X, False)

            if rank == 0:
                # rank = len(s)
                rank = len(s[s > 1e-10])

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

        return u, s, vh

    def approximate(self, rank=0):

        if rank == 0:
            rank = len(self.singvals)

        return (self.modes[:, :rank] * self.singvals[:rank]) @ self.time[:rank, :]
