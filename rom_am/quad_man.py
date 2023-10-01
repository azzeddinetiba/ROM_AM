import numpy as np
import scipy.linalg as sp
from rom_am.pod import POD
import sys
import warnings
from scipy.linalg import lstsq


class QUAD_MAN(POD):
    """
    Data-driven quadratic manifold class
    """

    def __init__(self):
        """
        Initialisation function of POD class

        :param snapshot: snapshot matrix
        :type snapshot: numpy.ndarray
        :param points_absciss: points abciss/coef to predict
        :type points_absciss: numpy.ndarray
        :param points_ordinate: points ordinate/base to predict
        :type points_ordinate: numpy.ndarray


        :returns: Instantiate a POD caller
        """

        super().__init__()
        self._n_mode = None
        self.Vr = None
        self._exp_var = None
        self._snapshot_reduce = None
        self._snapshot_reconstruct_linear = None
        self.W = None
        self.columns = None
        self.Wtilde = None
        self.Vbar = None
        self.err = None
        self._snapshot_reconstruct_quad = None
        self.variance_linear = None
        self.variance_quad = None
        self.s_pod = None

    def _kron_x_sq(self, Xmat):
        w = Xmat.T.shape[1]
        Xi = []
        for j in range(w):
            Xi.append(np.tile(Xmat.T[:, j].reshape(-1, 1),
                      (1, w - (j + 1) + 1)) * Xmat.T[:, j:w])

        X2 = np.concatenate(Xi[:], axis=1)

        return X2.T

    def decompose(self, X, alg="svd", rank=None, opt_trunc=False, tikhonov=0, thin=False,
                  column_selection=False, lbda=0, lbda_col=0.5, alpha0=0.01, error_comput=False):
        """
        Compute decomposition method of Quad_Manifold class

        :param min_exp_var: minimum cumulative explained energy required
        :type min_exp_var: float
        :param keepmode: choice of nb of modes that should be kept
        :type keepmode:
        :param energy_plot: plot the explained variance by all POD modes
        :type energy_plot: bool

        :returns: None
        :rtype: None
        """

        u, s, vh = super().decompose(X, alg, rank, opt_trunc, tikhonov, thin)

        self._snapshot_reduce = super().project(X)
        self._snapshot_reconstruct_linear = self.reconstruct()
        self.err = X - self._snapshot_reconstruct_linear

        # Compute the Kronecker product without redundancy
        self.W = self._kron_x_sq(self._snapshot_reduce)

        if column_selection:
            VbarTinit = np.zeros((self.W.T.shape[1], self.err.T.shape[1]))
            columns, _, _, _ = self._column_selection(
                self.W.T, self.err.T, VbarTinit, lbda_col, alpha0)
            self.columns = columns
            print("Colonnes retenues: ", columns)
            self.Wtilde = self.W[columns, :]
        else:
            self.columns = np.arange(0, self.W.shape[0], 1).astype(int)
            self.Wtilde = self.W

        q = self.err.shape[0]
        p = self.Wtilde.shape[0]
        Aplus = np.concatenate(
            (self.Wtilde.T, np.sqrt(lbda) * np.eye(p)), axis=0)
        bplus = np.concatenate((self.err.T, np.zeros((p, q))), axis=0)
        self.Vbar = lstsq(Aplus, bplus)[0].T

        self._snapshot_reconstruct_quad = self._snapshot_reconstruct_linear + \
            self.Vbar @ self.Wtilde

        # Compute error
        if error_comput:
            self.error = np.linalg.norm(self._snapshot_reconstruct_linear - X, ord='fro') / np.linalg.norm(X,
                                                                                                    ord='fro')
            self.error_quad = np.linalg.norm(self._snapshot_reconstruct_quad - X, ord='fro') / np.linalg.norm(
                X, ord='fro')
            print("Reconstruction error linear basis: ", self.error)
            print("Reconstruction error quadratic manifold: ", self.error_quad)

        return u, s, vh

    def inverse_project(self, new_reduced_data):

        return super().inverse_project(new_reduced_data) + self.Vbar @ self._kron_x_sq(new_reduced_data)

    def project(self, new_data):

        reduced_linear = super().project(new_data)
        return np.vstack((reduced_linear, self._kron_x_sq(reduced_linear)))

    def _column_selection(self, WT, EpsT, VbarT, lbda, alpha0):

        epstol = 1e-6
        convergence = 0
        itmax = 1000
        btmax = 50
        alpha_min = 1.e-20
        flag = 0

        r2 = WT.shape[1]
        n = EpsT.shape[1]

        VnewT = VbarT

        Temp = WT @ VbarT - EpsT
        Fval = 0.5 * np.trace(Temp.T @ Temp)
        for i in range(r2):
            Fval = Fval + lbda * np.linalg.norm(VbarT[i, :])

        alpha = alpha0

        for iter in range(itmax):
            G = WT @ VbarT - EpsT
            G = WT.T @ G
            for ia in range(btmax):
                R = VbarT - alpha * G
                zero_rows = 0
                for i in range(r2):
                    nRi = np.linalg.norm(R[i, :], ord=2)
                    if nRi <= alpha * lbda:
                        VnewT[i, :] = np.zeros((1, n))
                        zero_rows = zero_rows + 1
                    else:
                        VnewT[i, :] = (1.0 - (alpha * lbda) / nRi) * R[i, :]
                Temp = WT @ VnewT - EpsT
                Fval_new = 0.5 * np.trace(Temp.T @ Temp)
                for i in range(r2):
                    Fval_new = Fval_new + lbda * \
                        np.linalg.norm(VnewT[i, :], ord=2)
                if Fval_new < Fval:
                    print("Success with alpha= ", alpha)
                    break
                else:
                    alpha = 0.5 * alpha
                    if alpha < alpha_min:
                        print("alpha too small")
                        break
            if (ia == btmax) or (alpha < alpha_min):
                print("Backtracking failed")
                flag = 1
                break

            alpha = 1.5 * alpha
            if (np.abs(Fval_new - Fval) / np.abs(Fval) <= epstol):
                convergence = 1
            Fval = Fval_new
            VbarT = VnewT
            if convergence == 1:
                print("convergence - stop")
                break

        columns = []
        if convergence == 1:
            for i in range(r2):
                if (np.linalg.norm(VbarT[i, :]) > epstol):
                    columns.append(i)

        return columns, VbarT, alpha, flag


"""
    def predict(self, points_2_predict, rom_model="NI_POD", interp_method="linear_interp", save_interpolator=False,
                use_interpolator=False, opinf_modelform='A', opinf_reg=0.01):

        if rom_model == "NI_POD":
            u_pred = self.s_pod.predict(points_2_predict, interp_method=interp_method,
                                        save_interpolator=save_interpolator,
                                        use_interpolator=use_interpolator).T

            s_ref = np.repeat(self._snapshot_ref[:, 0].reshape(-1, 1), points_2_predict.shape[0], axis=1)
            W = self._kron_x_sq((self.Vr.T @ (u_pred - s_ref)).T).T
            Wtilde = W[self.columns, :]
            u_quad_pred = s_ref + u_pred + self.Vbar @ Wtilde

        elif rom_model == "OpInf":
            rom = opinf.DiscreteOpInfROM(modelform=opinf_modelform)
            rom.fit(basis=None, states=self._snapshot_reduce, regularizer=opinf_reg)
            shat_rom = rom.predict(state0=self._snapshot_reduce[:, 0], niters=points_2_predict.shape[0])
            s_ref = np.repeat(self._snapshot_ref[:, 0].reshape(-1, 1), points_2_predict.shape[0], axis=1)
            W = self._kron_x_sq((shat_rom).T).T
            Wtilde = W[self.columns, :]
            u_quad_pred = s_ref + self.Vr @ shat_rom + self.Vbar @ Wtilde

        return u_quad_pred  # , s_ref + self.Vr @ shat_rom
"""
