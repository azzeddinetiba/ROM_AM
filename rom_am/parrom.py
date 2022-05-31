import time
import numpy as np


class ParROM:
    """
    Parametric Non-Intrusive Reduced Order Modeling Class

    ...

    Parameters
    ----------
    parrom : python class
        an instance of a class that represents a method for reduced
        order modeling it has to have the methods decompose(),
        reconstruct() and predict()

        The class' decompose() method must take as arguments at least
        the {X, alg, params, rank1, rank2, opt_trunc, tikhonov} arguments, the predict()
        has to take at least {t, mu, t1, rank}

    """

    def __init__(self, parrom) -> None:

        self.model = parrom
        self.snapshots = None
        self.singvals = None
        self.modes = None
        self.time = None
        self.normalize = False
        self.profile = {}
        self.center = False
        self.norm_info = None
        self.Y = None
        self.Y_input = None
        self._accuracy = None

    def decompose(self,
                  X,
                  params,
                  alg="svd",
                  rank1=0,
                  rank2=0,
                  opt_trunc=False,
                  tikhonov=0,
                  center=False,
                  normalize=False,
                  normalization="norm",
                  norm_info=None,
                  *args,
                  **kwargs,):

        self.snapshots = X.copy()
        self.params = params.copy()
        if center:
            self.center = center
            self._center()

        if normalize:
            self.normalize = normalize
            self.norm_info = norm_info
            self.normalization = normalization
            if self.norm_info is not None:
                self.normalization = "spec"
            self._normalize()

        t0 = time.time()
        u, s, vh = self.model.decompose(X=self.snapshots,
                                        params=self.params,
                                        alg=alg,
                                        rank1=rank1,
                                        rank2=rank2,
                                        opt_trunc=opt_trunc,
                                        tikhonov=tikhonov,
                                        sorting="abs",
                                        *args,
                                        **kwargs,)

        self.singvals = s
        self.modes = u
        self.time = vh
        t1 = time.time()

        self.profile["Training time"] = t1-t0

    def _normalize(self, Y=None):
        """normalization of the input snapshots

        """
        if self.normalization == "minmax":
            self.snap_max = np.max(self.snapshots, axis=2).mean(axis=0)
            self.snap_min = np.min(self.snapshots, axis=2).mean(axis=0)
            self.max_min = (self.snap_max - self.snap_min)
            self.max_min = np.where(np.isclose(
                self.max_min, 0), 1, self.max_min)
            self.snapshots = (
                self.snapshots - self.snap_min[np.newaxis, :, np.newaxis]) / self.max_min[np.newaxis, :, np.newaxis]
        elif self.normalization == "norm":
            temp = self.snapshots
            self.snap_norms = np.linalg.norm(temp, axis=2)
            self.snap_norms = np.where(np.isclose(
                self.snap_norms, 0), 1, self.snap_norms).mean(axis=0)

            self.snapshots = self.snapshots / \
                self.snap_norms[np.newaxis, :, np.newaxis]

            self.param_norms = np.linalg.norm(self.params, axis=1)
            self.param_norms = np.where(np.isclose(
                self.param_norms, 0), 1, self.param_norms)

            self.params = self.params / \
                self.param_norms[:, np.newaxis]

        elif self.normalization == "spec":
            assert self.norm_info is not None, "Values for specific normalization are not assigned through the \
                'norm_info' argument"
            assert np.sum(self.norm_info[:, 1]) == self.snapshots.shape[1], "The sum of fields lengths (size of \
                second column of norm_info) should be the same as the size of input data"
            self.snapshots = self.snapshots / \
                np.repeat(self.norm_info[:, 0], self.norm_info[:, 1].astype(int))[
                    np.newaxis, :, np.newaxis]

    def _denormalize(self, res):
        """denormalization of the input array

        Parameters
        ----------
        res: numpy.ndarray, size (N, m)
            has the same axis 0 dimension as the input snapshots 

        Returns
        ----------
            numpy.ndarray, size (N, m)
            the denormalized array based on the min and max of the input snapshots
        """
        if self.normalization == "minmax":
            return res * self.max_min[:, np.newaxis] + self.snap_min[:, np.newaxis]
        elif self.normalization == "norm":
            return res * self.snap_norms[:, np.newaxis]
        elif self.normalization == "spec":
            return res * np.repeat(self.norm_info[:, 0], self.norm_info[:, 1].astype(int))[
                :, np.newaxis]

    def _center(self,):
        """Center the data along time

        """

        self.mean_flow = np.hstack(
            (self.snapshots, self.Y[:, -1].reshape((-1, 1)))).mean(axis=1)
        self.Y -= self.mean_flow.reshape((-1, 1))
        self.snapshots -= self.mean_flow.reshape((-1, 1))

    def _decenter(self, res):
        """Decenter the data on the input array

        Parameters
        ----------
        res: numpy.ndarray, size (N, m)
            has the same axis 0 dimension as the input snapshots 

        Returns
        ----------
            numpy.ndarray, size (N, m)
            the decentered data based on the mean of the input snapshots
        """
        return res + self.mean_flow.reshape((-1, 1))

    def predict(self, t, mu, t1=0, rank=None, *args, **kwargs):
        """Predict the solution of the reduced order model on the prescribed time instants and 
        the target aprameter value.

        Parameters
        ----------
        t : numpy.ndarray, size (nt, )
            time steps at which the ROM solution will be computed
        mu : float
            Parameter value for prediction
        t1: float
            the value of the time instant of the first snapshot
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None 

        Returns
        ----------
            numpy.ndarray, size (N, nt)
            ROM solution on the time values t
        """
        if self.normalize:
            mu = mu/self.param_norms[:, np.newaxis]
        t0 = time.time()
        res = self.model.predict(t=t, mu=mu, t1=t1, rank=rank, *args, **kwargs)
        t1 = time.time()
        if self.normalize:
            res = self._denormalize(res)
        if self.center:
            res = self._decenter(res)
        self.profile["Prediction time"] = t1-t0
        return res

    def get_accuracy(self, t, mu, ref, rank=None, t1=0):

        self._trained_on = self.snapshots
        err = np.linalg.norm(self.predict(
            t=t, mu=mu, rank=rank, t1=t1) - ref, axis=0)/np.linalg.norm(ref, axis=0)
        return err.sum()/err.shape[0]
