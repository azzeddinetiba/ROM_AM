import numpy as np
import scipy.linalg as sp
from rom_am import POD
import sys
import warnings
from scipy.linalg import lstsq

class QUAD_MAN:
    """
    Data-driven quadratic manifold class
    """

    def __init__(self, snapshot, points_absciss, points_ordinate, s_ref):
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

        self._snapshot_ref = s_ref
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
        self._snapshot = snapshot

    def _kron_x_sq(self, Xmat):
        w = Xmat.shape[1]
        Xi = []
        for j in range(w):
            Xi.append(np.tile(Xmat[:, j].reshape(-1, 1), (1, w - (j + 1) + 1)) * Xmat[:, j:w])

        X2 = np.concatenate(Xi[:], axis=1)

        return X2

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
                    Fval_new = Fval_new + lbda * np.linalg.norm(VnewT[i, :], ord=2)
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

    def compute_decomposition(
            self,
            min_exp_var=0.99,
            keepmode=3,
            column_selection=False, lbda=0, lbda_col=0.5, alpha0=0.01):

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

        # Find the POD basis : left singular vectors
        # U, Sigma, Vtranspose = svd(self._snapshot - self._snapshot_ref, full_matrices=False)
        #
        # cum_exp_var = np.cumsum(Sigma) / np.sum(Sigma)
        # self._exp_var = Sigma / np.sum(Sigma)
        # n_mode_var_exp = np.where(cum_exp_var > min_exp_var)[0][0] + 1
        # if keepmode == "exp_ener":
        #     self._n_mode = n_mode_var_exp
        # else:
        #     self._n_mode = keepmode
        # Vr_ref = U[:, :self._n_mode]

        s_pod = POD()
        s_pod.decompose((self._snapshot - self._snapshot_ref), rank = keepmode)
        self.s_pod = s_pod

        self._n_mode = self.s_pod.kept_rank
        # Un jour il faudrait quand même qu'on recode la classe POD en transposant la matrice de snapshot parce que
        # là on fait l'inverse de la littérature c'est quand même confusant
        self.Vr = self.s_pod.modes.copy()

        self._snapshot_reduce = self.Vr.T @ (self._snapshot - self._snapshot_ref)
        self._snapshot_reconstruct_linear = self._snapshot_ref + self.Vr @ self._snapshot_reduce

        self.err = self._snapshot - self._snapshot_ref - self.Vr @ self._snapshot_reduce
        # Compute the Kronecker product without redundancy
        self.W = self._kron_x_sq(self._snapshot_reduce.T).T
        if column_selection:
            VbarTinit = np.zeros((self.W.T.shape[1], self.err.T.shape[1]))
            columns, VbarT, alpha, flag = self._column_selection(self.W.T, self.err.T, VbarTinit, lbda_col, alpha0)
            self.columns = columns
            print("Colonnes retenues: ", columns)
            self.Wtilde = self.W[columns, :]
        else:
            self.columns = np.arange(0, self.W.shape[0], 1).astype(int)
            self.Wtilde = self.W

        q = self.err.shape[0]
        p = self.Wtilde.shape[0]
        Aplus = np.concatenate((self.Wtilde.T, np.sqrt(lbda) * np.eye(p)), axis=0)
        bplus = np.concatenate((self.err.T, np.zeros((p, q))), axis=0)
        self.Vbar = lstsq(Aplus, bplus)[0].T

        self._snapshot_reconstruct_quad = self._snapshot_ref + self.Vr @ self._snapshot_reduce + \
                                          self.Vbar @ self.Wtilde

        # Compute error
        err = np.linalg.norm(self._snapshot_reconstruct_linear - self._snapshot, ord='fro') / np.linalg.norm(self._snapshot,
                                                                                                   ord='fro')
        err_quad = np.linalg.norm(self._snapshot_reconstruct_quad - self._snapshot, ord='fro') / np.linalg.norm(
            self._snapshot, ord='fro')
        print("Reconstruction error linear basis: ", err)
        print("Reconstruction error quadratic manifold: ", err_quad)

        # Retained variance
        self.variance_linear = np.linalg.norm(self._snapshot_reconstruct_linear, ord='fro') / np.linalg.norm(self._snapshot,
                                                                                                   ord='fro')
        self.variance_quad = np.linalg.norm(self._snapshot_reconstruct_quad, ord='fro') / np.linalg.norm(
            self._snapshot, ord='fro')
        print("Retained variance on the linear POD basis: ", self.variance_linear * 100.0)
        print("Retained variance on the quadratic manifold: ", self.variance_quad * 100.0)

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
