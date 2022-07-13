from re import A
import numpy as np
from .pod import POD
import warnings


class DMD:
    """
    Dynamic Mode Decomposition Class

    """

    def __init__(self):

        self.singvals = None
        self.modes = None
        self.time = None
        self.dmd_modes = None
        self.dt = None
        self.t1 = None
        self.n_timesteps = None
        self.tikhonov = None
        self.x_cond = None
        self._kept_rank = None
        self.init = None
        self.A_tilde = None
        self._A = None
        self._no_reduction = False
        self.pred_rank = None
        self._pod_coeff = None
        self.stock = False
        self.data = None
        self._koop_eigv = None
        self._koop_modes = None
        self._left_eigenvectors = None
        self._low_dim_left_eig = None

    def decompose(self,
                  X,
                  alg="svd",
                  rank=None,
                  opt_trunc=False,
                  tikhonov=0,
                  sorting="abs",
                  Y=None,
                  dt=None,
                  no_reduc=False,
                  stock=False):
        """Training the dynamic mode decomposition[1] model, using the input data X and Y

        Parameters
        ----------
        X : numpy.ndarray
            Snapshot matrix data, of (N, m) size
        Y : numpy.ndarray
            Second Snapshot matrix data, of (N, m) size
            advanced from X by dt
        dt : float
            value of time step from each snapshot in X
            to each snapshot in Y
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
        stock: bool, optional
            Whteher to store the data snapshots in one of the object attributes;
            dmd.data
            Default False

        References
        ----------

        [1] On dynamic mode decomposition:  Theory and applications,
        Journal of Computational Dynamics,1,2,391,421,2014-12-1,
        Jonathan H. Tu,Clarence W. Rowley,Dirk M. Luchtenburg,
        Steven L. Brunton,J. Nathan Kutz,2158-2491_2014_2_391,

        [2] M. Gavish and D. L. Donoho, "The Optimal Hard Threshold for
        Singular Values is 4/sqrt(3) ," in IEEE Transactions on Information
        Theory, vol. 60, no. 8, pp. 5040-5053, Aug. 2014,
        doi: 10.1109/TIT.2014.2323359.

        Returns
        ------
        u : numpy.ndarray, of size(N, r)
            The spatial modes of the training data

        s : numpy.ndarray, of size(r, )
            The singular values of the training data

        vh : numpy.ndarray, of size(r, m)
            The time dynamics of the training data


        """
        if X.shape[0] != Y.shape[0]:
            raise Exception("The DMD input snapshots X and Y have to have \
                the same observables, for different observables in X and Y, \
                    Please consider using the EDMD() class")
        if X.shape[1] != Y.shape[1]:
            raise Exception("The DMD input snapshots X and Y have to have \
                the same number of instants")

        self.tikhonov = tikhonov
        if self.tikhonov:
            self.x_cond = np.linalg.cond(X)

        self.n_timesteps = X.shape[1]
        self.init = X[:, 0]
        self.sorting = sorting

        if stock:
            self.stock = True
            self.data = np.hstack((X, Y[:, -1].reshape((-1, 1))))

        if not no_reduc:
            # POD Decomposition of the X matrix
            self.pod_ = POD()
            self.pod_.decompose(X, alg=alg, rank=rank,
                                opt_trunc=opt_trunc)
            u = self.pod_.modes
            vh = self.pod_.time
            s = self.pod_.singvals
            self._kept_rank = self.pod_.kept_rank

            # Computing the A_tilde: the projection of the 'A' operator
            # on the POD modes, where A = Y * pseudoinverse(X) [1]
            s_inv = np.zeros(s.shape)
            s_inv = 1 / s
            if self.tikhonov:
                s_inv *= s**2 / (s**2 + self.tikhonov * self.x_cond)
            store = np.linalg.multi_dot((Y, vh.T, np.diag(s_inv)))
            self.A_tilde = u.T @ store

            # Eigendecomposition on the low dimensional operator
            lambd, w = np.linalg.eig(self.A_tilde)
            if sorting == "abs":
                idx = (np.abs(lambd)).argsort()[::-1]
            else:
                idx = (np.real(lambd)).argsort()[::-1]
            lambd = lambd[idx]
            w = w[:, idx]
            self.low_dim_eig = w

            # Computing the high-dimensional DMD modes [1]
            phi = store @ w
        else:  # Should ONLY be used in the context of very low dimensional data
            self._no_reduction = True
            self._A = Y @ np.linalg.pinv(X)
            lambd, phi = np.linalg.eig(self._A)

            s = 0.
            u = 0.
            vh = 0.

        omega = np.log(lambd) / dt  # Continuous system eigenvalues

        # Loading the DMD instance's attributes
        self.dt = dt
        self.dmd_modes = phi
        self._koop_modes = self.dmd_modes
        self.lambd = lambd
        self.eigenvalues = omega
        self.singvals = s
        self.modes = u
        self.time = vh

        return u, s, vh

    def predict(self, t, t1=0, method=0, rank=None, stabilize=True, init=None, given_b=False, cutoff=1.):
        """Predict the DMD solution on the prescribed time instants.

        Parameters
        ----------
        t : numpy.ndarray, size (nt, )
            time steps at which the DMD solution will be computed
        t1: float
            the value of the time instant of the first data snapshot
            If 'method=1' is used, t1 indicates the time instant 
            when the solution corresponds to 'init'
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None
        method: int
            Method used to compute the initial mode amplitudes
            0 if it is computed on the POD subspace as in Tu et al.[1] (Least Expensive)
            1 if it is computed using the pseudoinverse of the DMD modes along
                the initial snapshots
            2 if it is computed as a least square fit for the DMD modes along
                all snapshots (Most expensive)
            Default : 0
        stabilize : bool, optional
            DMD eigenvalue-shifting to stable eigenvalues at the prediction
            phase
            Default : True
        init : int or ndarray or None, optional
            The initial condition used to compute the amplitudes
            it is an :
                - int when used with 'method = 0', representing the index
                of the snapshot used from the snapshots training data.
                Note that here t1 will be taken as the time insant at that
                snapshot (init * dt), so the t1 argument is here the time
                instant of the first snapshot data.
                - ndarray of size (n, ) when used with 'method = 1',
                representing the prescribed initial condition at t = t1 (It has
                to be prescribed accordingly).
                - None, then the first data snapshot will be used
                whether in 'method = 0' or 'method = 1'
                - Disregarded when used with 'method = 3'
            Default None
        References
        ----------

        [1] On dynamic mode decomposition:  Theory and applications,
        Journal of Computational Dynamics,1,2,391,421,2014-12-1,
        Jonathan H. Tu,Clarence W. Rowley,Dirk M. Luchtenburg,
        Steven L. Brunton,J. Nathan Kutz,2158-2491_2014_2_391,

        Returns
        ----------
            numpy.ndarray, size (N, nt)
            DMD solution on the time values t
        """
        if self.dmd_modes is None:
            raise Exception("The DMD decomposition hasn't been executed yet")

        if rank is None:
            rank = self._kept_rank
        elif not (isinstance(rank, int) and 0 < rank < self._kept_rank):
            warnings.warn("The rank chosen for prediction should be an integer smaller than the "
                          "rank chosen/computed at the decomposition phase. Please see the rank value in self.kept_rank")
            rank = self._kept_rank

        if not given_b:
            b = self._compute_amplitudes(
                method, rank=self.pred_rank, initial=init)

        eig = self.eigenvalues[:rank]
        if stabilize:
            eig_rmpl = eig[np.abs(self.lambd[:rank]) > cutoff]
            eig_rmpl.real = 0
            eig[np.abs(self.lambd[:rank]) > cutoff] = eig_rmpl

        self.t1 = t1
        if method == 0:
            if init is None:
                init = 0.
            t1 = self.t1 + init * self.dt
        # ================================================================
        # When using method = 2, the low dimensional DMD modes W
        # were used for the amplitudes computations (least square problem)
        # The High dimensional DMD modes however are the one used for the
        # final prediction computation. Those DMD modes are computed here
        # using (exact modes) phi = Y @ V @ (1/Sig) @ W and not (projected
        # modes) phi = U @ W, so the high-dimensional DMD modes are mapped
        # to the Y range and not the X range, so we have to adjust the
        # initial time instant value to be the second instant value.
        # ================================================================
        if method == 2:
            t1 = self.t1 + self.dt

        return self.dmd_modes[:, :rank] @ (np.exp(np.outer(eig, t-t1).T) * self.computed_amplitudes[:rank]).T

    def reconstruct(self, rank=None):
        """Reconstruct the data input using the DMD Model.

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
            DMD solution on the time steps where the input snapshots are taken
        """
        if rank is None:
            rank = self._kept_rank
        elif not (isinstance(rank, int) and 0 < rank < self.kept_rank):
            warnings.warn("The rank chosen for reconstruction should be an integer smaller than the "
                          "rank chosen/computed at the decomposition phase. Please see the rank value in self.kept_rank")
            rank = self._kept_rank

        if self.t1 is None:
            self.t1 = 0
            warnings.warn("the initial instant value was not assigned during the prediction phase, "
                          "t1 is chosen as 0")

        t = np.linspace(self.t1 + self.dt, self.t1 + self.n_timesteps
                        * self.dt, self.n_timesteps)
        return self.predict(t, t1=self.t1)

    def _compute_amplitudes(self, method, rank=None, initial=None):
        """Predict the DMD solution on the prescribed time instants.

        Parameters
        ----------

        method: int
            Method used to compute the initial mode amplitudes
            0 if it is computed on the POD subspace as in Tu et al.[1] (Least Expensive)
            1 if it is computed using the pseudoinverse of the DMD modes along
                the initial snapshots
            2 if it is computed as a least square fit for the DMD modes along
                all snapshots (Most expensive) [2]
        rank: int or None
            ranks kept for prediction: it should be a hard threshold integer
            and greater than the rank chose/computed in the decomposition
            phase. If None, the same rank already computed is used
            Default : None
        initial : int or ndarray or None, optional
            The initial condition used to compute the amplitudes
            it is an :
                - int when used with 'method = 0', representing the index
                of the snapshot used from the snapshots training data.
                Note that here t1 will be taken as the time insant at that
                snapshot (init * dt), so the t1 argument is here the time
                instant of the first snapshot data.
                - ndarray of size (n, ) when used with 'method = 1',
                representing the prescribed initial condition at t = t1 (It has
                to be prescribed accordingly).
                - None, then the first data snapshot will be used
                whether in 'method = 0' or 'method = 1'

        References
        ----------

        [1] On dynamic mode decomposition:  Theory and applications,
        Journal of Computational Dynamics,1,2,391,421,2014-12-1,
        Jonathan H. Tu,Clarence W. Rowley,Dirk M. Luchtenburg,
        Steven L. Brunton,J. Nathan Kutz,2158-2491_2014_2_391,

        [2] Higher Order Dynamic Mode Decompositio
        Soledad Le Clainche and José M. Vega
        SIAM Journal on Applied Dynamical Systems 2017 16:2, 882-925

        Returns
        ----------
            numpy.ndarray, size (N, nt)
            DMD solution on the time values t
        """
        if method == 1:
            if initial is None:
                init = self.init
            else:
                init = initial
            try:
                b, _, _, _ = np.linalg.lstsq(self.dmd_modes, init, rcond=None)
            except:
                raise Exception(
                    "The init argument should be an ndarray or None when 'method=1' is used")
        elif method == 2:
            if rank is None:
                rank = self._kept_rank
            else:
                rank = rank
            if self.data is None:
                n_steps = self.n_timesteps - 1
            else:
                n_steps = self.n_timesteps

            # ======= The L matrix contains [W; WΛ; WΛ**2; ... ; WΛ**(m-1)]
            # ======== Meaning L is of size (rm x r)
            n_modes = len(self.lambd)
            L = self.low_dim_eig[:rank, :] @ np.tile(np.eye(n_modes), n_steps) * \
                np.tile(self.lambd, n_steps)**np.repeat(
                np.linspace(1, n_steps, n_steps, dtype=int), n_modes)
            L = np.vstack((self.low_dim_eig[:rank, :], L.reshape(
                rank, -1, n_modes).swapaxes(0, 1).reshape((-1, n_modes))))
            # ========== Solving the lstsq system L b = v
            # =========== where v are the pod coefficients of snapshots (i.e of size(rm, ))
            b, _, _, _ = np.linalg.lstsq(
                L, (self.pod_coeff[:rank, :]).reshape((-1, 1), order='F').ravel(), rcond=None)
        elif method == 3:
            if initial is None:
                init = self.init
            else:
                init = initial
            try:
                b = self.koop_eigf(init[:, np.newaxis])[:, 0]
            except:
                raise Exception(
                    "The init argument should be an ndarray or None when 'method=3' is used")
        else:
            if initial is None:
                # The default choice is using the first snapshot's POD coefficients
                initial = 0
            try:
                alpha1 = self.singvals * self.time[:, initial]
            except:
                raise Exception(
                    "init argument should be an int or None when 'method=0' is used")
            b = np.linalg.solve(self.lambd * self.low_dim_eig, alpha1)

        self.computed_amplitudes = b
        return b

    @property
    def A(self):
        """Computes the high dimensional DMD operator.

        """
        if self._A is None:
            self._A = self.modes @ self.A_tilde @ self.modes.T
        return self._A

    @property
    def pod_coeff(self):

        if self._pod_coeff is None:
            if self.data is not None:
                self._pod_coeff = self.modes.T @ self.data
            else:
                self._pod_coeff = np.diag(self.singvals) @ self.time
        return self._pod_coeff

    @property
    def koop_modes(self):
        """Returns the koopman modes.

        """
        try:
            if self._koop_modes is None:
                self._koop_modes = self.dmd_modes
            return self._koop_modes
        except AttributeError:
            raise AttributeError("DMD Decomposition is not yet computed")

    @property
    def koop_eigv(self):
        """Returns the koopman eigenvalues.

        """
        try:
            if self._koop_eigv is None:
                self._koop_eigv = self.lambd
            return self._koop_eigv
        except AttributeError:
            raise AttributeError("DMD Decomposition is not yet computed")

    @property
    def low_dim_left_eig(self):
        """Returns the reduced left eigenvectors

        """
        if self._low_dim_left_eig is None:
            # Left Eigendecomposition on the low dimensional operator
            lambd, w = np.linalg.eig(self.A_tilde.T)
            if self.sorting == "abs":
                idx = (np.abs(lambd)).argsort()[::-1]
            else:
                idx = (np.real(lambd)).argsort()[::-1]
            idx = (np.abs(lambd)).argsort()[::-1]
            lambd = lambd[idx]
            w = w[:, idx]

            # Left eigenvectors scaled so as w . v = 1.
            inpdct = w.T @ self.low_dim_eig
            scl = np.diag(inpdct)
            scl = scl/np.abs(scl)
            self._low_dim_left_eig = w/np.linalg.norm(inpdct, axis=0) * scl

        return self._low_dim_left_eig

    @property
    def left_eigvectors(self):
        """Returns the left eigenvectors of the DMD operator.

        """
        if self._left_eigenvectors is None:
            self._left_eigenvectors = self.modes @ self.low_dim_left_eig

        return self._left_eigenvectors

    def koop_eigf(self, x):
        """Computes the Koopman eigenfunction at x

        Parameters
        ----------

        x: ndarray, of shape (N, nt)
            m points of N dimension at which eigenfunctions will
            be computes
            N must be the same as the dimension of snapshots

        Returns
        ----------
            numpy.ndarray, size (k, nt)
            the k eigenfunctions computed at the x points

        """

        return self.left_eigvectors.T @ x
