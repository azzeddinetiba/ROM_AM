import time
import numpy as np


class ROM:
    """
    Non-Intrusive Reduced Order Modeling Class

    ...

    Parameters
    ----------
    rom_object : python class
        an instance of a class that represents a method for reduced
        order modeling it has to have the methods decompose(),
        reconstruct() and predict()

        The class' decompose() method must take as arguments at least
        the same arguments of ROM.decompose(), same thing for
        ROM.reconstruct() and ROM.predict()

    """

    def __init__(self, rom_object):

        self.model = rom_object
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

    def decompose(
            self,
            X,
            alg="svd",
            rank=0,
            opt_trunc=False,
            tikhonov=0,
            center=False,
            normalize=False,
            normalization="norm",
            norm_info=None,
            *args,
            **kwargs,):
        """Computes the data decomposition, training the model on the input data X.
                                            (SVD - based)

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
        center : bool, optional
            Flag to either center the data around 0 or not
            Default : False
        normalize : bool, optional
            Flag to either normalize the data or not
            Default : False
        normalization : str, optional
            The type of normalization used : "norm" for normalization by
            the L2 norm or "minmax" for the min-max normalization or "spec"
            for specific field normalization (division of each field by a
            specific value), using "spec" flag should be accompanied by the
            'norm_info' argument
            Default : "norm"
        norm_info : numpy.ndarray 2D
            a 2D numpy array containing the value of norms each field will 
            be divided on in the first column, the second column contains 
            the sizeof the field accoring to each value, the sum of the
            second column should be equal to the size of input data

        References
        ----------

        [1] On dynamic mode decomposition:  Theory and applications,
        Journal of Computational Dynamics,1,2,391,421,2014-12-1,
        Jonathan H. Tu,Clarence W. Rowley,Dirk M. Luchtenburg,
        Steven L. Brunton,J. Nathan Kutz,2158-2491_2014_2_391,



        """

        self.snapshots = X.copy()
        if "Y" in kwargs.keys():
            self.Y = kwargs["Y"].copy()
            if "Y_input" in kwargs.keys():
                self.Y_input = kwargs["Y_input"].copy()
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
        if "Y" in kwargs.keys():
            kwargs["Y"] = self.Y
            if "Y_input" in kwargs.keys():
                kwargs["Y_input"] = self.Y_input

        t0 = time.time()
        u, s, vh = self.model.decompose(X=self.snapshots,
                                        alg=alg,
                                        rank=rank,
                                        opt_trunc=opt_trunc,
                                        tikhonov=tikhonov,
                                        *args,
                                        **kwargs,)

        self.singvals = s
        self.modes = u
        self.time = vh
        t1 = time.time()

        self.profile["Training time"] = t1-t0

    def predict(self, t, t1=0, rank=None, *args, **kwargs):
        """Predict the solution of the reduced order model on the prescribed time instants.

        Parameters
        ----------
        t : numpy.ndarray, size (nt, )
            time steps at which the ROM solution will be computed
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
        t0 = time.time()
        res = self.model.predict(t=t, t1=t1, rank=rank, *args, **kwargs)
        if self.normalize:
            res = self._denormalize(res)
        if self.center:
            res = self._decenter(res)
        t1 = time.time()
        self.profile["Prediction time"] = t1-t0
        return res

    def reconstruct(self, rank=None):
        """Reconstruct the data input using the Reduced Order Model.

        Parameters
        ----------
        rank: int or None
            ranks kept for reconstruction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None 

        Returns
        ----------
            numpy.ndarray, size (N, m)
            ROM solution on the time steps where the input snapshots are taken
        """
        return self.model.reconstruct(rank=rank)

    def _normalize(self, Y=None):
        """normalization of the input snapshots

        """
        if self.normalization == "minmax":
            if self.Y is None:
                self.snap_max = np.max(self.snapshots, axis=1)
                self.snap_min = np.min(self.snapshots, axis=1)
                self.max_min = ((self.snap_max - self.snap_min)[:, np.newaxis])
                self.max_min = np.where(np.isclose(
                    self.max_min, 0), 1, self.max_min)
            else:
                self.snap_max = np.max(
                    np.hstack((self.snapshots, self.Y[:, -1].reshape((-1, 1)))), axis=1)
                self.snap_min = np.min(
                    np.hstack((self.snapshots, self.Y[:, -1].reshape((-1, 1)))), axis=1)
                self.max_min = ((self.snap_max - self.snap_min)[:, np.newaxis])
                self.max_min = np.where(np.isclose(
                    self.max_min, 0), 1, self.max_min)
                self.Y = (self.Y - self.snap_min[:, np.newaxis]) / self.max_min
            self.snapshots = (
                self.snapshots - self.snap_min[:, np.newaxis]) / self.max_min
        elif self.normalization == "norm":
            if self.Y is None:
                temp = self.snapshots
            elif self.Y_input is None:
                temp = np.hstack(
                    (self.snapshots, self.Y[:, -1].reshape((-1, 1))))
            else:
                temp = np.vstack(
                    (np.hstack((self.snapshots, self.Y[:, -1].reshape((-1, 1)))), np.hstack((self.Y_input, self.Y_input[:, -1][:, np.newaxis]))))
            self.snap_norms = np.linalg.norm(temp, axis=1)
            self.snap_norms = np.where(np.isclose(
                self.snap_norms, 0), 1, self.snap_norms)
            if self.Y_input is not None:
                self.Y_input = self.Y_input / \
                    self.snap_norms[-1, np.newaxis]
                self.Y = self.Y / self.snap_norms[:-1, np.newaxis]
                self.snapshots = self.snapshots / \
                    self.snap_norms[:-1, np.newaxis]
            else:
                self.snapshots = self.snapshots / \
                    self.snap_norms[:, np.newaxis]
                if self.Y is not None:
                    self.Y = self.Y / self.snap_norms[:, np.newaxis]
        elif self.normalization == "spec":
            assert self.norm_info is not None, "Values for specific normalization are not assigned through the \
                'norm_info' argument"
            assert np.sum(self.norm_info[:, 1]) == self.snapshots.shape[0], "The sum of fields lengths (size of \
                second column of norm_info) should be the same as the size of input data"
            if self.Y is not None:
                self.Y = self.Y / \
                    np.repeat(self.norm_info[:, 0], self.norm_info[:, 1].astype(int))[
                        :, np.newaxis]
            self.snapshots = self.snapshots / \
                np.repeat(self.norm_info[:, 0], self.norm_info[:, 1].astype(int))[
                    :, np.newaxis]

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
            return res * self.max_min + self.snap_min[:, np.newaxis]
        elif self.normalization == "norm":
            if self.Y_input is not None:
                return res * self.snap_norms[:-1, np.newaxis]
            else:
                return res * self.snap_norms[:, np.newaxis]
        elif self.normalization == "spec":
            return res * np.repeat(self.norm_info[:, 0], self.norm_info[:, 1].astype(int))[
                :, np.newaxis]

    def _center(self,):
        """Center the data along time

        """
        if self.Y is None:
            self.mean_flow = self.snapshots.mean(axis=1)
        else:
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

    def reconstruct(self, rank=None):
        return self.model.reconstruct(rank)

    @property
    def accuracy(self, rank=None, t=None, ref=None, t1=0):
        """Gives the accuracy of the ROM, compared to the input data or to reference values

        Parameters
        ----------
        rank: int
            ranks kept for prediction/reconstruction: it should be a hard
            threshold integer and greater than the rank chose/computed in
            the decomposition phase. If None, the same rank already computed
            is used
            Default : None
        t: ndarray (m, )
            The time instants of the ROM solution to be compared to the
            reference
            in case it is not assigned, the accuracy is copmputed for
            reconstruction (compared to the input snapshots)
        ref: ndarray (N, m)
            The reference solution to which the ROM solution is compared,
            in case it is not assigned, the accuracy is copmputed for
            reconstruction (compared to the input snapshots)
            has the same axis 1 dimension as the axis 0 in the 't' argument
        t1: float
            the value of the time instant of the first snapshot

        Returns
        ----------
            float
            the relative error of the ROM
        """
        if t is None:
            err = np.linalg.norm(self.reconstruct(
                rank=rank) - self.snapshots, axis=0)/np.linalg.norm(self.snapshots, axis=0)
            return err.sum()/err.shape[0]
        else:
            err = np.linalg.norm(self.predict(
                t=t, rank=rank, t1=t1) - ref, axis=0)/np.linalg.norm(ref, axis=0)
            return err.sum()/err.shape[0]
