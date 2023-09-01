import numpy as np
from numpy import newaxis
from scipy.interpolate import RBFInterpolator
from rom_am import ROM, POD, QUAD_MAN
import time
from sklearn.linear_model import Ridge, LassoLarsIC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import copy
import torchvision
import torch
from .inferring_model import AutEnc


class solid_ROM:

    def __init__(self, method=0):
        self.project_time = np.array([])
        self.regression_time = np.array([])
        self.inverse_project_time = np.array([])
        self.map_mat = None
        self.inverse_project_mat = None
        self.stored_disp_coeffs = []
        self.ridge = False
        self.lasso = False
        self.norm_regr = False
        self.ranks_pres = None
        self.coords_extracted = False
        self.current_disp_coeff = None
        self.torch_model = None

    def train(self,
              pres_data,
              disp_data,
              kernel='thin_plate_spline',
              rank_disp=None,
              rank_pres=None,
              epsilon=1.,
              ids=None,
              ids_regr=None,
              map_used=None,
              ridge=False,
              lasso=False,
              norm_regr=True,
              degree=3,
              alpha=1e-5,
              quad_=False,
              torch_model=None,
              norm=["minmax", "minmax"]):

        # ========= Separation of final iterations and subiterations ==================
        unused_disp_data = None
        m = disp_data.shape[1]
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
        disp_opt_trunc = True
        if rank_disp is not None:
            disp_opt_trunc = False
        if quad_:
            disp_pod = QUAD_MAN()
        else:
            disp_pod = POD()
        disp_rom = ROM(disp_pod)
        disp_rom.decompose(X=used_disp_data, normalize=True, center=True,
                           opt_trunc=disp_opt_trunc, rank=rank_disp)
        disp_coeff = disp_pod.pod_coeff
        self.disp_pod = disp_pod
        self.disp_rom = disp_rom
        if ids is not None:
            normal_unused_disp_data = disp_rom.normalize(
                disp_rom.center(unused_disp_data))

        self.rank_disp = self.disp_pod.kept_rank

        # ========= Reduction of the pressure data ==================
        if torch_model is None:

            pres_opt_trunc = True
            if rank_pres is not None:
                pres_opt_trunc = False
            pres_pod = POD()
            pres_rom = ROM(pres_pod)
            pres_rom.decompose(X=used_pres_data, normalize=True, center=True,
                               opt_trunc=pres_opt_trunc, rank=rank_pres)
            pres_coeff = pres_pod.pod_coeff
            self.pres_rom = pres_rom
            self.pres_pod = pres_pod
            if ids is not None:
                normal_unused_pres_data = pres_rom.normalize(
                    pres_rom.center(unused_pres_data))

                unus_pres_coeff = self.pres_pod.project(
                    normal_unused_pres_data)
                unus_disp_coeff = self.disp_pod.project(
                    normal_unused_disp_data)
                if quad_:
                    unus_disp_coeff = unus_disp_coeff[:self.rank_disp, :]

            self.rank_pres = self.pres_pod.kept_rank

        else:
            self.torch_model = True
            pres_coeff = pres_data

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
        if ids_regr is None:
            if ids is None:
                pres_coeff_tr = pres_coeff.copy()
                disp_coeff_tr = disp_coeff.copy()
            else:
                pres_coeff_tr = np.hstack((unus_pres_coeff, pres_coeff))
                disp_coeff_tr = np.hstack(
                    (unus_disp_coeff, disp_coeff))

            if ridge and torch_model is None:
                self.ridge = True

                model_regr = make_pipeline(
                    PolynomialFeatures(degree), Ridge(alpha=alpha))
                model_regr.fit(pres_coeff_tr.T, disp_coeff_tr.T)
                self.model_regr = model_regr

            elif lasso and torch_model is None:
                self.lasso = True

                model_regr = make_pipeline(
                    PolynomialFeatures(degree), MultiOutputRegressor(LassoLarsIC(criterion='bic')))
                model_regr.fit(pres_coeff_tr.T, disp_coeff_tr.T)
                self.model_regr = model_regr

            elif (not ridge and not lasso) and torch_model is None:
                self.func = [RBFInterpolator(
                    pres_coeff_tr.T, disp_coeff_tr.T, kernel=kernel, epsilon=epsilon, degree=degree)]
            else:
                self.infer_model = AutEnc()
                self.infer_model.load_state_dict(torch.load(torch_model))
                self.infer_model.eval()
                self.infer_model = self.infer_model.double()

        else:

            all_coeff_pres = np.empty(
                (pres_pod.kept_rank, pres_data.shape[1]))
            all_coeff_disp = np.empty(
                (disp_pod.kept_rank, disp_data.shape[1]))

            if ids is not None:
                all_coeff_pres[:, ids] = pres_coeff
                all_coeff_disp[:, ids] = disp_coeff

                all_coeff_pres[:, np.setdiff1d(
                    np.arange(0, m, 1), ids)] = unus_pres_coeff
                all_coeff_disp[:, np.setdiff1d(
                    np.arange(0, m, 1), ids)] = unus_disp_coeff
            else:
                all_coeff_pres = pres_coeff.copy()
                all_coeff_disp = disp_coeff.copy()

            self.func = []
            for i in range(len(ids_regr)):
                pres_coeff_tr = all_coeff_pres[:, ids_regr[i]]
                disp_coeff_tr = all_coeff_disp[:, ids_regr[i]]
                if not ridge and not lasso:
                    used_kernel = kernel
                    if type(kernel) is list:
                        used_kernel = kernel[i]

                    self.func.append(RBFInterpolator(
                        pres_coeff_tr.T, disp_coeff_tr.T, kernel=used_kernel, epsilon=epsilon))
                else:
                    self.ridge = True

                    model_regr = make_pipeline(
                        PolynomialFeatures(degree), Ridge(alpha=alpha))
                    model_regr.fit(pres_coeff_tr.T, disp_coeff_tr.T)
                    self.func.append(copy.deepcopy(model_regr))

        self.saved_prs_cf_tr = pres_coeff_tr.copy()
        self.saved_disp_cf_tr = disp_coeff_tr.copy()

        if map_used is not None:
            self.map_mat = map_used
            self.inverse_project_mat = self.map_mat @ self.disp_rom.denormalize(
                self.disp_pod.modes)

            if quad_:
                self.inverse_project_Vbar = self.map_mat @ self.disp_rom.denormalize(
                    self.disp_pod.Vbar)

            if self.disp_rom.center:
                self.mapped_mean_flow = self.map_mat @ self.disp_rom.mean_flow.reshape(
                    (-1, 1))

    def pred(self, new_pres, which_func=None):

        if which_func is None:
            which_func = 0

        t0 = 0.
        if not self.torch_model:

            interm = (
                new_pres-self.pres_rom.mean_flow[:, np.newaxis])/self.pres_rom.snap_norms[:, np.newaxis]
            t0 = time.time()
            pred_pres_coeff = self.pres_pod.project(interm)

        else:
            pred_pres_coeff = new_pres

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
        if (not self.ridge and not self.lasso) and self.torch_model is None:
            res1 = self.func[which_func](pred_pres_coeff.T).T
        elif (self.ridge or self.lasso) and self.torch_model is None:
            res1 = self.model_regr.predict(pred_pres_coeff.T).T

        else:
            new_load = torch.tensor(pred_pres_coeff.T).double()
            v_latent_ = self.infer_model.encoder(new_load)
            res1 = self.infer_model.infer(
                self.infer_model.poly_terms(v_latent_)).detach().numpy().T
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
        if self.map_mat is None:
            res2 = self.disp_pod.inverse_project(res1)
            res = self.disp_rom.decenter(self.disp_rom.denormalize(res2))
            t5 = time.time()

        else:

            self.current_disp_coeff = res1.copy()

            try:
                res = (self.inverse_project_mat @ res1 +
                       self.inverse_project_Vbar @ self.disp_pod._kron_x_sq(res1))
            except AttributeError:
                res = (self.inverse_project_mat @ res1)
            t5 = time.time()

            res = (res + self.mapped_mean_flow).reshape((-1, 2))

        # ============== Saving some profiling information =====================
        if len(self.project_time) < 21:
            self.project_time = np.append(self.project_time, t1-t0)
            self.regression_time = np.append(self.regression_time, t3-t2)
            self.inverse_project_time = np.append(
                self.inverse_project_time, t5-t4)

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

        res_interm = self.disp_pod.modes @ np.hstack(self.stored_disp_coeffs)
        try:
            res_interm += self.disp_pod.Vbar @ self.disp_pod._kron_x_sq(
                np.hstack(self.stored_disp_coeffs))
        except AttributeError:
            pass

        return self.disp_rom.decenter(self.disp_rom.denormalize(res_interm))
