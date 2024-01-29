import numpy as np
import time
from rom_am.regressors.polynomialLassoRegressor import PolynomialLassoRegressor
from rom_am.regressors.polynomialLassoDynamicalRegressor import PolynomialLassoDynamicalRegressor
from rom_am.regressors.dynamicalRbfRegressor import DynamicalRBFRegressor
from rom_am.regressors.polynomialRegressor import PolynomialRegressor
from rom_am.regressors.rbfRegressor import RBFRegressor
from rom_am.dimreducers.rom_am.podReducer import PodReducer
from rom_am.dimreducers.rom_am.quadManReducer import QuadManReducer
import pickle


class solid_ROM:

    def __init__(self, is_dynamical=False):
        self.encoding_time = np.array([])
        self.regression_time = np.array([])
        self.decoding_time = np.array([])
        self.map_mat = None
        self.inverse_project_mat = None
        self.stored_disp_coeffs = []
        self.norm_regr = False
        self.current_disp_coeff = None
        self.torch_model = None
        if is_dynamical:
            self.is_dynamical = True
        else:
            self.is_dynamical = False

    def train(self,
              pres_data,
              disp_data,
              rank_disp=None,
              rank_pres=None,
              ids=None,
              map_used=None,
              regression_model=None,
              forcesReduc_model=None,
              dispReduc_model=None,
              norm_regr=[False, False],
              norm=["l2", "l2"],
              algs=["svd", "svd"],
              to_copy=[True, True],
              remove_outliers=False,
              previous_disp_data=None):
        """Training the solid ROM model

        Parameters
        ----------
        pres_data  : numpy.ndarray
            Snapshot matrix of fluid forces data (input), of (Nf, m) size
        disp_data  : numpy.ndarray
            Snapshot matrix of fluid forces data (output), of (Ns, m) size
        rank_disp  : int, double or None
            The latent dimension for the displacement field.
            If dispReduc_model is a an instance of a custom class, rank_disp
            is ignored. If dispReduc_model is chosen among `dimreducers`
            implemented classes, it is chosen as the latent dimension.
            If None, no reduction is used on the POD.
            Default : None
        rank_pres  : int, double or None
            The latent dimension for the forces field.
            If forcesReduc_model is a an instance of a custom class,
            `rank_pres` is ignored. If forcesReduc_model is chosen
            among `dimreducers` implemented classes, it is chosen as
            the latent dimension.
            If None, no reduction is used on the POD.
            Default : None
        ids        : numpy.ndarray, optional
            Array of indices corresponding to the data points used for the
            dimensionality reduction. If None, all the data is used.
            Default : None
        map_used   : numpy.ndarray or None
            Snapshot matrix of mapping indices (from interface
            nodes to all the nodes), of (N, n) size.
            If None, no mapping is used
            Default : None
        regression_model   : str, RomRegressor or None
            An instance of the RomRegressor class for the regression.
            If None or "PolyLasso", PolynomialLassoRegressor is used.
            If "PolyRidge" PolynomialRegressor is used.
            If "RBF" RBFRegressor is used.
            Default : None
        forcesReduc_model   : str, RomDimensionalityReducer or None
            An instance of the RomDimensionalityReducer class for the input encoder.
            If None or "POD", PodReducer is used.
            Default : None
        dispReduc_model   : str, RomDimensionalityReducer or None
            An instance of the RomDimensionalityReducer class for the output decoder.
            If None or "QUAD", QuadManReducer is used.
            If "POD", PodReducer is used
            Default : None
        norm_regr  : list of 2 booleans, optional
            Whether to normalize the inputs and outputs of the regression
            model. The normalization used is chosen by the `norm`argument
            Default : [True, True]
        norm       : list of 2 strs, optional
            Type of normalization used ([inputs, outputs]).
            "minmax" for min-max normalization. "l2" for L2
            normalization. "std" for standardization.
            Default : ["minmax", "minmax"]
        norm       : list of 2 strs, optional
            Type of decomposition used ([inputs, outputs]).
            Whether to use the SVD on decomposition ("svd") or
            the eigenvalue problem on snaphot matrices ("snap")
            Default : ["svd", "svd"]

        Returns
        ------

        """

        # ========= Separation of converged iterations and subiterations ==================
        unused_disp_data = None
        m = disp_data.shape[1]
        self.map_mat = map_used
        assert (m == pres_data.shape[1])
        if ids is None:
            used_disp_data = disp_data
            used_pres_data = pres_data

        else:
            used_disp_data = disp_data[:, ids]
            unused_disp_data = disp_data[:,
                                         np.setdiff1d(np.arange(0, m, 1), ids)]
            used_pres_data = pres_data[:, ids]
            unused_pres_data = pres_data[:,
                                         np.setdiff1d(np.arange(0, m, 1), ids)]

        # ========= Reduction of the displacement data ==================
        if dispReduc_model == "POD":
            self.dispReduc_model = PodReducer(latent_dim=rank_disp)
        elif dispReduc_model is None or dispReduc_model == "QUAD":
            self.dispReduc_model = QuadManReducer(
                latent_dim=rank_disp)
        else:
            self.dispReduc_model = dispReduc_model

        self.dispReduc_model.train(used_disp_data, map_used=map_used, alg = algs[1], to_copy=to_copy[1])
        disp_coeff = self.dispReduc_model.reduced_data

        if self.is_dynamical:
            previous_disp_coeff = self.dispReduc_model.encode(previous_disp_data)
        else:
            previous_disp_coeff = None

        # ========= Reduction of the pressure data ==================
        if forcesReduc_model is None or forcesReduc_model == "POD":
            self.forcesReduc = PodReducer(latent_dim=rank_pres)
        elif forcesReduc_model == "QUAD":
            self.forcesReduc = QuadManReducer(latent_dim=rank_pres)
        else:
            self.forcesReduc = forcesReduc_model

        self.forcesReduc.train(used_pres_data, alg = algs[0], to_copy=to_copy[0])
        pres_coeff = self.forcesReduc.reduced_data

        if ids is not None:

            self.forcesReduc.check_encoder_in(unused_pres_data)
            unus_pres_coeff = self.forcesReduc.encode(
                unused_pres_data)
            self.forcesReduc.check_encoder_out(unus_pres_coeff)
            self.dispReduc_model.check_encoder_in(unused_disp_data)
            unus_disp_coeff = self.dispReduc_model.encode(
                unused_disp_data)
            self.dispReduc_model.check_encoder_out(unus_disp_coeff)

        self.norm_regr = [False, False]
        self.norms = norm
        if norm_regr[1]:
            if norm[1] == "minmax":
                disp_coeff_max = np.max(disp_coeff, axis=1)[:, np.newaxis]
                disp_coeff_min = np.min(disp_coeff, axis=1)[:, np.newaxis]
                disp_coeff_max_min = np.where(np.isclose(
                    disp_coeff_max - disp_coeff_min, 0), 1, disp_coeff_max - disp_coeff_min)

                disp_coeff = (disp_coeff - disp_coeff_min) / \
                    (disp_coeff_max_min)
                if self.is_dynamical:
                    previous_disp_coeff = (previous_disp_coeff - disp_coeff_min) / \
                        (disp_coeff_max_min)

                self.disp_coeff_max = disp_coeff_max
                self.disp_coeff_min = disp_coeff_min
                self.disp_coeff_max_min = disp_coeff_max_min
            elif norm[1] == "l2":
                disp_coeff_mean = np.mean(disp_coeff, axis=1)[:, np.newaxis]
                disp_coeff_nrm = np.linalg.norm(
                    disp_coeff, axis=1)[:, np.newaxis]

                disp_coeff = (disp_coeff - disp_coeff_mean) / \
                    (disp_coeff_nrm)
                if self.is_dynamical:
                    previous_disp_coeff = (previous_disp_coeff - disp_coeff_mean) / \
                        (disp_coeff_nrm)

                self.disp_coeff_mean = disp_coeff_mean
                self.disp_coeff_nrm = disp_coeff_nrm
            elif norm[1] == "std":
                disp_coeff_mean = np.mean(disp_coeff, axis=1)[:, np.newaxis]
                disp_coeff_std = np.std(disp_coeff, axis=1)[:, np.newaxis]

                disp_coeff = (disp_coeff - disp_coeff_mean) / \
                    (disp_coeff_std)
                if self.is_dynamical:
                    previous_disp_coeff = (previous_disp_coeff - disp_coeff_mean) / \
                        (disp_coeff_std)

                self.disp_coeff_mean = disp_coeff_mean
                self.disp_coeff_std = disp_coeff_std

            self.norm_regr[1] = True

        if norm_regr[0]:
            if norm[0] == "minmax":
                pres_coeff_max = np.max(pres_coeff, axis=1)[:, np.newaxis]
                pres_coeff_min = np.min(pres_coeff, axis=1)[:, np.newaxis]
                pres_coeff_max_min = np.where(np.isclose(
                    pres_coeff_max - pres_coeff_min, 0), 1, pres_coeff_max - pres_coeff_min)

                pres_coeff = (pres_coeff - pres_coeff_min) / \
                    (pres_coeff_max_min)

                self.pres_coeff_max = pres_coeff_max
                self.pres_coeff_min = pres_coeff_min
                self.pres_coeff_max_min = pres_coeff_max_min
            elif norm[0] == "l2":
                pres_coeff_mean = np.mean(pres_coeff, axis=1)[:, np.newaxis]
                pres_coeff_nrm = np.linalg.norm(
                    pres_coeff, axis=1)[:, np.newaxis]

                pres_coeff = (pres_coeff - pres_coeff_mean) / (pres_coeff_nrm)

                self.pres_coeff_nrm = pres_coeff_nrm
                self.pres_coeff_mean = pres_coeff_mean
            elif norm[0] == "std":
                pres_coeff_mean = np.mean(pres_coeff, axis=1)[:, np.newaxis]
                pres_coeff_std = np.std(pres_coeff, axis=1)[:, np.newaxis]

                pres_coeff = (pres_coeff - pres_coeff_mean) / (pres_coeff_std)

                self.pres_coeff_std = pres_coeff_std
                self.pres_coeff_mean = pres_coeff_mean

            self.norm_regr[0] = True

            if ids is not None:
                unus_disp_coeff = (unus_disp_coeff - disp_coeff_min) / \
                    (disp_coeff_max_min)
                unus_pres_coeff = (unus_pres_coeff -
                                   pres_coeff_min) / (pres_coeff_max_min)

        # self.saved_pres_coeff = pres_coeff.copy()
        # self.saved_disp_coeff = diameter_coeff.copy()

        # ========= Regression =========
        if ids is None:
            pres_coeff_tr = pres_coeff
            disp_coeff_tr = disp_coeff
        else:
            pres_coeff_tr = np.hstack((unus_pres_coeff, pres_coeff))
            disp_coeff_tr = np.hstack(
                (unus_disp_coeff, disp_coeff))

        # Removing outliers
        if remove_outliers:
            pres_coeff_tr, ids_tr = self.reject_outliers(pres_coeff_tr)
            disp_coeff_tr = disp_coeff_tr[:, ids_tr]
            if self.is_dynamical:
                previous_disp_coeff = previous_disp_coeff[:, ids_tr]

        if not self.is_dynamical:
            if regression_model is None or regression_model == "PolyLasso":
                self.regressor = PolynomialLassoRegressor(
                    poly_degree=2, criterion='bic')
            elif regression_model == "PolyRidge":
                self.regressor = PolynomialRegressor()
            elif regression_model == "RBF":
                self.regressor = RBFRegressor()
            else:
                self.regressor = regression_model
        else:
            if regression_model is None or regression_model == "PolyDynamicalLasso":
                self.regressor = PolynomialLassoDynamicalRegressor(
                    poly_degree=2, criterion='bic')
            elif regression_model == "RBF":
                self.regressor = DynamicalRBFRegressor()
            else:
                self.regressor = regression_model

        if not self.is_dynamical:
            self.regressor.train(pres_coeff_tr, disp_coeff_tr)
        else:
            self.regressor.train(pres_coeff_tr, disp_coeff_tr, previous_disp_coeff)
        # self.saved_prs_cf_tr = pres_coeff_tr.copy()
        # self.saved_disp_cf_tr = disp_coeff_tr.copy()

    def reject_outliers(self, data, m=8):
        ids_ = np.max((np.abs(data - np.mean(data, axis=1).reshape((-1, 1)))), axis = 0) < m*np.std(data, axis = 0)
        return data[:, ids_], ids_

    def pred(self, new_pres, previous_disp=None):
        """Solid ROM prediction

        Parameters
        ----------
        new_pres  : numpy.ndarray
            Snapshot matrix of input data, of (Nin, m) size

        Returns
        ------
        output_result : numpy.ndarray
            Solution matrix data, of (Nout, m) size

        """
        t0 = time.time()
        self.forcesReduc.check_encoder_in(new_pres)
        pred_pres_coeff = self.forcesReduc.encode(new_pres)
        if pred_pres_coeff is None:
            return None
        else:
            self.forcesReduc.check_encoder_out(pred_pres_coeff)
        t1 = time.time()

        if self.norm_regr[0]:
            if self.norms[0] == "minmax":
                pred_pres_coeff = (pred_pres_coeff -
                                   self.pres_coeff_min) / self.pres_coeff_max_min
            elif self.norms[0] == "l2":
                pred_pres_coeff = (pred_pres_coeff -
                                   self.pres_coeff_mean) / self.pres_coeff_nrm
            elif self.norms[0] == "std":
                pred_pres_coeff = (pred_pres_coeff -
                                   self.pres_coeff_mean) / self.pres_coeff_std

        if self.is_dynamical:
            self.dispReduc_model.check_encoder_in(previous_disp)
            previous_disp_coeff = self.dispReduc_model.encode(previous_disp)
            self.dispReduc_model.check_encoder_out(previous_disp_coeff)
            if self.norm_regr[1]:
                if self.norms[0] == "minmax":
                    previous_disp_coeff = (previous_disp_coeff -
                                    self.disp_coeff_min) / self.disp_coeff_max_min
                elif self.norms[0] == "l2":
                    previous_disp_coeff = (previous_disp_coeff -
                                    self.disp_coeff_mean) / self.disp_coeff_nrm
                elif self.norms[0] == "std":
                    previous_disp_coeff = (previous_disp_coeff -
                                    self.disp_coeff_mean) / self.disp_coeff_std

        # self.saved_pres_pred_coeff = pred_pres_coeff.copy()
        # ============== Regression predicts =====================
        t2 = time.time()
        self.regressor.check_predict_in(pred_pres_coeff)
        if not self.is_dynamical:
            res1 = self.regressor.predict(pred_pres_coeff)
        else:
            res1 = self.regressor.predict(pred_pres_coeff, previous_disp_coeff)
        self.regressor.check_predict_out(res1)
        t3 = time.time()

        # ============== Denormalize Displ. coefficients =====================
        if self.norm_regr[1]:
            if self.norms[1] == "minmax":
                res1 = (res1 * self.disp_coeff_max_min) + self.disp_coeff_min
            elif self.norms[1] == "l2":
                res1 = (res1 * self.disp_coeff_nrm) + self.disp_coeff_mean
            elif self.norms[1] == "std":
                res1 = (res1 * self.disp_coeff_std) + self.disp_coeff_mean

        t4 = time.time()
        self.dispReduc_model.check_decoder_in(res1)
        res = self.dispReduc_model.decode(res1)
        self.dispReduc_model.check_decoder_out(res)
        if self.map_mat is not None:
            self.current_disp_coeff = res1.copy()
        t5 = time.time()

        # ============== Saving some profiling information =====================
        if len(self.encoding_time) < 21:
            self.encoding_time = np.append(self.encoding_time, t1-t0)
            self.regression_time = np.append(self.regression_time, t3-t2)
            self.decoding_time = np.append(
                self.decoding_time, t5-t4)

        return res

    def save_times(self, files_names):

        if len(self.encoding_time) < 21:
            time_arrays = [self.encoding_time, self.regression_time,
                           self.decoding_time,]
            for i in range(3):
                np.save(files_names[i], time_arrays[i])
        else:
            pass

    def store_last_result(self):

        if self.current_disp_coeff is not None:
            self.stored_disp_coeffs.append(self.current_disp_coeff.copy())

    def return_big_disps(self):
        return self.dispReduc_model.decode(np.hstack(self.stored_disp_coeffs), high_dim=True)

    def save(self, file_name):
        with open(file_name+'.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
