import numpy as np
import scipy.linalg as sp
import sys
import warnings

os_ = 1
if "linux" in sys.platform:
    os_ = 0
    import jax.numpy as jnp
    import jax

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)


class POD:
    """
    Proper Orthogonal Decomposition Class

    """

    def __init__(self):
        self.kept_rank = None
        self.singvals = None
        self.modes = None
        self.time = None

    def decompose(self, X, alg="svd", rank=0, opt_trunc=False, tikhonov=0, thin=False):
        """Computes the proper orthogonal decomposition, training the model on the input data X.

        Parameters
        ----------
        X : numpy.ndarray
            Snapshot matrix data, of (N, m) size
        alg : str, optional
            Whether to use the SVD on decomposition ("svd") or
            the eigenvalue problem on snaphot matrices ("snap")
            Default : "svd"
        rank : int or float, optional
            if rank = 0 All the ranks are kept, unless their
            singular values are zero
            if 0 < rank < 1, it is used as the percentage of
            the energy that should be kept, and the rank is
            computed accordingly
            Default : 0
        opt_trunc : bool, optional
            if True an optimal truncation/threshold is estimated,
            based on the algorithm of Gavish and Donoho [1]
            Default : False
        tikhonov : int or float, optional
            tikhonov parameter for regularization
            If 0, no regularization is applied, if float, it is used as
            the lambda tikhonov parameter
            Default : 0

        References
        ----------

        [1] On dynamic mode decomposition:  Theory and applications,
        Journal of Computational Dynamics,1,2,391,421,2014-12-1,
        Jonathan H. Tu,Clarence W. Rowley,Dirk M. Luchtenburg,
        Steven L. Brunton,J. Nathan Kutz,2158-2491_2014_2_391,


        """
        min_dim = min(X.shape[0], X.shape[1])
        if rank < 0 or (rank > 1 and not isinstance(rank, int) and
                        not isinstance(rank, np.int64) and
                        not isinstance(rank, np.int32)):
            raise ValueError("Invalid rank value, it should be an integer greater "
                             "than 0 or a float between 0 and 1")
        if rank > min_dim:
            warnings.warn("The rank chosen for reconstruction should not be greater than "
                          "the smallest data dimension m, the rank is now chosen as m")
            rank = min_dim

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
                    beta = X.shape[0]/X.shape[1]
                else:
                    beta = X.shape[1]/X.shape[0]
                omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
                tau = np.median(s) * omega
                rank = np.sum(s > tau)
            else:
                if rank == 0:
                    if thin:
                        rank = min_dim
                    else:
                        rank = len(s[s > 1e-10])
                elif 0 < rank < 1:
                    rank = np.searchsorted(
                        np.cumsum(s**2 / (s**2).sum()), rank) + 1
            self.kept_rank = rank

            u = u[:, :rank]
            vh = vh[:rank, :]
            s = s[:rank]

        elif alg == "snap":
            cov = X.T @ X
            lambd, v = np.linalg.eigh(cov)

            lambd = np.flip(lambd)
            v = np.flip(v, 1)
            lambd[lambd < 1e-10] = 0
            s = np.sqrt(lambd)

            if opt_trunc:
                if X.shape[0] <= X.shape[1]:
                    beta = X.shape[0]/X.shape[1]
                else:
                    beta = X.shape[1]/X.shape[0]
                omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
                tau = np.median(s) * omega
                rank = np.sum(s > tau)
            else:
                if rank == 0:
                    if thin:
                        rank = min_dim
                    else:
                        rank = len(lambd[lambd > 1e-10])
                elif 0 < rank < 1:
                    rank = np.searchsorted(
                        np.cumsum(s**2 / (s**2).sum()), rank) + 1

            s = s[:rank]
            s_inv = np.zeros(s.shape)
            s_inv[s > 1e-10] = 1.0 / s[s > 1e-10]
            v = v[:, :rank]
            vh = v.T

            u = X @ v[:, :rank] * s_inv

        self.singvals = s
        self.modes = u
        self.time = vh

        self.tikhonov = tikhonov

        return u, s, vh

    def reconstruct(self, rank=None):
        """Reconstruct the data input using the POD Model.

        Parameters
        ----------
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None 

        Returns
        ----------
            numpy.ndarray, size (N, m)
            POD Reconstruction on the time steps where the input snapshots are taken
        """
        if self.singvals is None:
            raise Exception("The POD decomposition hasn't been executed yet")

        if rank is None:
            rank = self.kept_rank
        elif not (isinstance(rank, int) and 0 < rank < self.kept_rank):
            warnings.warn("The rank chosen for reconstruction should be an integer smaller than the "
                          "rank chosen/computed at the decomposition phase. Please see the rank value by self.kept_rank")
            rank = self.kept_rank

        return (self.modes[:, :rank] * self.singvals[:rank]) @ self.time[:rank, :]
