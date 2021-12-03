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

<<<<<<< HEAD
<<<<<<< Updated upstream
    def decompose(self, X, Y=None, Y_input=None, dt=None, center=False, alg="svd", rank=0):
=======
    def decompose(self, X, Y=None, dt=None, center=False, alg="svd", rank=0):
>>>>>>> Stashed changes
=======
    def decompose(self, X, Y=None, Y_input=None, dt=None, center=False, alg="svd", rank=0, sorting="abs"):
>>>>>>> a40d9b6b992b4c2c265886754f220a7a0e4e562d

        if center:
            self.mean_flow = X.mean(axis=1)
            X -= self.mean_flow.reshape((-1, 1))

        if self.rom == "pod":
            u, s, vh = self._pod_decompose(X, alg, rank)

        else:
            if self.rom == "dmd":
                u, s, vh, lambd, phi = self._dmd_decompose(
                    X, Y, rank, sorting=sorting)
            elif self.rom == "dmdc":
                u_til_1, u_til_2, s_til, vh_til, lambd, phi = self._dmdc_decompose(
                    X, Y, Y_input, rank)
                u = u_til_1
                vh = vh_til
                s = s_til

            omega = np.log(lambd) / dt

            self.dmd_modes = phi
            self.lambd = lambd
            self.eigenvalues = omega

        self.singvals = s
        self.modes = u
        self.time = vh

    def _dmd_decompose(self, X, Y, rank=0, sorting="abs"):

        if os_ == 0:
            u, s, vh = jnp.linalg.svd(X, False)
            u = np.array(u)
            s = np.array(s)
            vh = np.array(vh)
        else:
            u, s, vh = sp.svd(X, False)

        if rank == 0:
            # rank = len(s)
            rank = len(s[s > 1e-10])

        u = u[:, :rank]
        vh = vh[:rank, :]
        s = s[:rank]

        s_inv = np.zeros(s.shape)
        s_inv[s > 1e-10] = 1 / s[s > 1e-10]
        store = np.linalg.multi_dot((Y, vh.T, np.diag(s_inv)))
        self.A_tilde = u.T @ store

        lambd, w = np.linalg.eig(self.A_tilde)

        if sorting == "abs":
            idx = (np.abs(lambd)).argsort()[::-1]
        else:
            idx = (np.real(lambd)).argsort()[::-1]
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        phi = store @ w

        return u, s, vh, lambd, phi

    def _dmdc_decompose(self, X, Y, Y_input, rank_til=0, rank_hat=0):

        omega = np.vstack((X, Y_input))
        if os_ == 0:
            jax.config.update("jax_enable_x64", True)
            u_til, s_til, vh_til = jnp.linalg.svd(omega, False)
            u_hat, s_hat, _ = jnp.linalg.svd(Y, False)
            u_til = np.array(u_til)
            s_til = np.array(s_til)
            vh_til = np.array(vh_til)
            u_hat = np.array(u_hat)
            s_hat = np.array(s_hat)

        else:
            u_til, s_til, vh_til = sp.svd(omega, False)
            u_hat, s_hat, _ = sp.svd(Y, False)

        if rank_til == 0:
            # rank = len(s)
            rank_til = len(s_til[s_til > 1e-10])
            rank_hat = len(s_hat[s_hat > 1e-10])

        u_til = u_til[:, :rank_til]
        vh_til = vh_til[:rank_til, :]
        s_til = s_til[:rank_til]
        u_til_1 = u_til[:X.shape[0], :]
        u_til_2 = u_til[:Y_input.shape[0], :]

        u_hat = u_hat[:, :rank_hat]

        s_inv = np.zeros(s_til.shape)
        s_inv[s_til > 1e-10] = 1 / s_til[s_til > 1e-10]

        store_ = np.linalg.multi_dot((Y, vh_til.T, np.diag(s_inv)))
        store = np.linalg.multi_dot((u_hat.T, store_))
        self.A_tilde = np.linalg.multi_dot((store, u_til_1.T, u_hat))
        self.B_tilde = np.linalg.multi_dot((store, u_til_2.T))

        lambd, w = np.linalg.eig(self.A_tilde)
        idx = np.abs(np.imag(lambd)).argsort()
        lambd = lambd[idx]
        w = w[:, idx]
        self.low_dim_eig = w

        phi = np.linalg.multi_dot((store_, u_til_1.T, u_hat, w))

        return u_til_1, u_til_2, s_til, vh_til, lambd, phi

    def _pod_decompose(self, X, alg, rank=0):

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
                u = np.array(u)
                s = np.array(s)
                vh = np.array(vh)
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

    def dmd_predict(self, t, init):

        # b, _, _, _ = np.linalg.lstsq(self.dmd_modes, init, rcond=None)
        alpha1 = self.singvals * self.time[:, 0]
        b = np.linalg.solve(self.lambd * self.low_dim_eig, alpha1)

        return self.dmd_modes @ (np.exp(np.outer(self.eigenvalues, t).T) * b).T
