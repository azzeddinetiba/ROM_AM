import numpy as np
from scipy.spatial import KDTree
from rom_am.pod import POD
from rom_am.dmd import DMD
from rom_am.hodmd import HODMD
from copy import deepcopy


class KnnDMD:

    def __init__(self) -> None:
        pass

    def decompose(self,
                  X,
                  params,
                  alg="svd",
                  rank=None,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  dt=None,
                  dmd_model="dmd",
                  hod=50):
        """Training the Knn dynamic mode decomposition model 
        using the input data X and the training parameters params

        Parameters
        ----------
        X : numpy.ndarray
            Parametric snapshot matrix data, of (p, N, m) size
        params : numpy.ndarray
            Parameters in a (k, p) array
        dt : float
            value of time step from each column in X to the next column
        alg : str, optional
            Whether to use the SVD on decomposition ("svd") or
            the eigenvalue problem on snaphot matrices ("snap")
            Default : "svd"
        rank : None, int or float, optional
            if rank = 0 or rank is None All the ranks are kept, 
            unless their singular values are zero
            if 0 < rank < 1, it is used as the percentage of
            the energy that should be kept, and the rank is
            computed accordingly
            Default : None
        rank2 : None, int or float, optional
            Rank chosen for the truncation of the POD coefficients
            inside the DMD algrithm.
            if rank = 0 All the ranks are kept, unless their
            singular values are zero
            if 0 < rank < 1, it is used as the percentage of
            the energy that should be kept, and the rank is
            computed accordingly
            Default : None
        opt_trunc : bool, optional
            if True an optimal truncation/threshold is estimated,
            based on the algorithm of Gavish and Donoho [2]
            Default : False
        tikhonov : int or float, optional
            tikhonov parameter for regularization
            If 0, no regularization is applied, if float, it is used as
            the lambda tikhonov parameter
            Default : 0
        sorting : str, optional
            Whether to sort the discrete DMD eigenvalues by absolute
            value ("abs") or by their real part ("real")
            Default : "abs"
        """

        self._p = X.shape[0]  # Number of parameters samples
        self._N = X.shape[1]  # Number of dofs
        self._m = X.shape[2]  # Number of timesteps

        self.param_tree = KDTree(params.T)

        self.stacked_X = X.swapaxes(0, 2).swapaxes(
            0, 1).reshape((self._N, self._m*self._p), order='F')  # of size (N, m * p)
        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(self.stacked_X)

        self.params = params
        self._k = params.shape[0]

        self.dmd_model = []
        for i in range(self._p):

            if dmd_model == "dmd":
                # DMD Decomposition
                tmp_model = DMD()
                _, _, _ = tmp_model.decompose(X[i, :, :-1], Y=X[i, :, 1::],
                                              alg=alg, dt=dt, rank=rank, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting)
            elif dmd_model == "hodmd":
                # High-Order DMD Decomposition
                tmp_model = HODMD()
                _, _, _ = tmp_model.decompose(X[i, :, :-1], Y=X[i, :, 1::],
                                              alg=alg, dt=dt, rank=rank, opt_trunc=opt_trunc, tikhonov=tikhonov, sorting=sorting, hod=hod)

            self.dmd_model.append(deepcopy(tmp_model))

        return tmp_model.modes, tmp_model.singvals, tmp_model.time

    def predict(self, t, mu, t1, rank=None, stabilize=False, init=None, method=0, k=None):
        """Predict the knndmd solution on the prescribed time instants and 
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
        if k is None:
            k = self._k + 1

        _, ii = self.param_tree.query(mu.ravel(), k=k)

        weights = np.linalg.norm(mu - self.params[:, ii], axis=0)
        weights /= weights.sum()

        sample_res = np.empty(
            (k, self._N, t.shape[0]), dtype=complex)

        for i in range(k):
            sample_res[i, :, :] = self.dmd_model[ii[i]].predict(
                t=t, t1=t1, method=method, rank=rank, stabilize=stabilize, init=init)

        weighted_res = np.dot(sample_res.T, weights).T

        return weighted_res
