import numpy as np
from rom_am import ROM, POD, QUAD_MAN
import time
import torchvision
import torch
# from .inferring_model import AutEnc
from rom_am.regressors.polynomialLassoRegressor import PolynomialLassoRegressor
from rom_am.regressors.polynomialRegressor import PolynomialRegressor
from rom_am.regressors.rbfRegressor import RBFRegressor
from rom_am.dimreducers.rom_am.podReducer import PodReducer
from rom_am.dimreducers.rom_am.quadManReducer import QuadManReducer


class solid_ROM:

    def __init__(self,):
        self.encoding_time = np.array([])
        self.regression_time = np.array([])
        self.decoding_time = np.array([])
        self.map_mat = None
        self.inverse_project_mat = None
        self.stored_disp_coeffs = []
        self.norm_regr = False
        self.current_disp_coeff = None
        self.torch_model = None

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
              norm_regr=[True, True],
              norm=["minmax", "minmax"]):

        # ========= Separation of converged iterations and subiterations ==================
        unused_disp_data = None
        m = disp_data.shape[1]
        self.map_mat = map_used
        assert (m == pres_data.shape[1])
        if ids is None:
            used_disp_data = disp_data.copy()
            used_pres_data = pres_data.copy()

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
        elif dispReduc_model is None or forcesReduc_model == "QUAD":
            self.dispReduc_model = QuadManReducer(
                latent_dim=rank_disp)
        else:
            self.dispReduc_model = dispReduc_model

        self.dispReduc_model.train(used_disp_data, map_used=map_used)
        disp_coeff = self.dispReduc_model.reduced_data

        # ========= Reduction of the pressure data ==================
        if forcesReduc_model is None or forcesReduc_model == "POD":
            self.forcesReduc = PodReducer(latent_dim=rank_pres)
        elif forcesReduc_model == "QUAD":
            self.forcesReduc = QuadManReducer(latent_dim=rank_pres)
        else:
            self.forcesReduc = forcesReduc_model

        self.forcesReduc.train(used_pres_data)
        pres_coeff = self.forcesReduc.reduced_data

        if ids is not None:

            unus_pres_coeff = self.forcesReduc.encode(
                unused_pres_data)
            unus_disp_coeff = self.dispReduc_model.encode(
                unused_disp_data)

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

                self.disp_coeff_max = disp_coeff_max
                self.disp_coeff_min = disp_coeff_min
                self.disp_coeff_max_min = disp_coeff_max_min
            elif norm[1] == "l2":
                disp_coeff_mean = np.mean(disp_coeff, axis=1)[:, np.newaxis]
                disp_coeff_nrm = np.linalg.norm(
                    disp_coeff, axis=1)[:, np.newaxis]

                disp_coeff = (disp_coeff - disp_coeff_mean) / \
                    (disp_coeff_nrm)

                self.disp_coeff_mean = disp_coeff_mean
                self.disp_coeff_nrm = disp_coeff_nrm
            elif norm[1] == "std":
                disp_coeff_mean = np.mean(disp_coeff, axis=1)[:, np.newaxis]
                disp_coeff_std = np.std(disp_coeff, axis=1)[:, np.newaxis]

                disp_coeff = (disp_coeff - disp_coeff_mean) / \
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

        if regression_model is None or regression_model == "PolyLasso":
            self.regressor = PolynomialLassoRegressor(
                poly_degree=2, criterion='bic')
        elif regression_model == "PolyRdige":
            self.regressor = PolynomialRegressor()
        elif regression_model == "RBF":
            self.regressor = RBFRegressor()
        else:
            self.regressor = regression_model

        self.regressor.train(pres_coeff_tr, disp_coeff_tr)
        # self.saved_prs_cf_tr = pres_coeff_tr.copy()
        # self.saved_disp_cf_tr = disp_coeff_tr.copy()

    def pred(self, new_pres):

        t0 = time.time()
        pred_pres_coeff = self.forcesReduc.encode(new_pres)
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

        # self.saved_pres_pred_coeff = pred_pres_coeff.copy()
        # ============== Regression predicts =====================
        t2 = time.time()
        res1 = self.regressor.predict(pred_pres_coeff)
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
        res = self.dispReduc_model.decode(res1)
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

        if len(self.project_time) < 21:
            time_arrays = [self.project_time, self.regression_time,
                           self.inverse_project_time,]
            for i in range(3):
                np.save(files_names[i], time_arrays[i])
        else:
            pass

    def store_last_result(self):

        if self.current_disp_coeff is not None:
            self.stored_disp_coeffs.append(self.current_disp_coeff.copy())

    def return_big_disps(self):
        return self.dispReduc_model.decode(np.hstack(self.stored_disp_coeffs), high_dim=True)
