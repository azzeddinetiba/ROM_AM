import numpy as np
import collections
from rom_am.regressors.polynomialRegressor import PolynomialRegressor
from rom_am.regressors.polynomialLassoRegressor import PolynomialLassoRegressor
from rom_am.regressors.skNNRegressor import SKNNRegressor
from rom_am.dimreducers.rom_DimensionalityReducer import RomDimensionalityReducer
from rom_am.dimreducers.rom_am.podReducer import PodReducer
from rom_am.dimreducers.rom_am.manifReducer import ManifInterpReducer
from rom_am.regressors.rbfRegressor import RBFRegressor
from rom_am.regressors.polynomialDynamicalRegressor import PolynomialDynamicalRegressor
import pickle
import copy
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
from rom_am.pod import POD
from rom_am.utils import rank1_update, _determine_pod_alg_square_matrices
from rom_am.rpod import _compute_past
from scipy.linalg import subspace_angles
import time
from joblib import Parallel, delayed


class TrackedFluidSurrog:

    def __init__(self, maxLen=6900, reTrainThres=240, parametric=False, updateBasis=False, updateThres=300,
                 updateOmega=False, njobs_online=1, alg_square_matrices=None):
        self.trainIn = collections.deque(maxlen=maxLen)
        self.trainOut = collections.deque(maxlen=maxLen)
        self.maxLen = maxLen
        self.countAugment = 0
        self.countUpdate = 0
        self.reTrainThres = reTrainThres
        self.retrain_count = 0
        self.retrain_times = []
        self.reTrainKernel = None
        self.reTrainSmoothing = None
        self._disp_latent_dim = None
        self._load_latent_dim = None
        self.param_encoder = None
        self.param_decoder = None
        self.param_regressor = None
        self.multiple_regressor = None
        self.single_regressor = None
        self._p = None
        self._predictedBasis = False
        self.hidden_layers = None
        self.new_regressor = None
        self.omega0 = 0.5
        self._omega_terms = (1-self.omega0, self.omega0)
        self.updateBasis = updateBasis
        self.updateThres = updateThres
        self.sendSignalBasis = None
        self.recursive_angles = []
        self.updateOmega = updateOmega
        self.retrainingTime = []
        self.stacked_calib = None
        self.njobs_online = 1
        self.alg_square_matrices = alg_square_matrices

    def sigmoid(self, x, n=1):
        s = int(n/2)
        a = 0.1*n
        e = np.exp(-(x-s)/a)
        m = (1. + e)
        return 1/m

    def train(self, dispData, fluidPrevData, fluidData, kernel='thin_plate_spline', smoothing=9.5e-2,
              rank_pres=.9999, rank_disp=.9999, degree=2, solidReduc: RomDimensionalityReducer = None, epsilon=1.,
              norm=[True, True], center=[True, True], norm_regr="max", params=None, weights=None, param_encoder=False,
              param_decoder=False, param_regressor=False, multiple_param_regressor=False, hidden_layers=np.array([
            40, 40, 40]), normalization=["norm", "norm"], initAlpha=6e-3, cleanup=True, alg="svd"):

        self.single_regressor = True
        warning_text = "The data matrix should be of shape (p, N, m) with p the number of parameters"
        if multiple_param_regressor:
            self._p = params.shape[1]
            self.multiple_regressor = True
            assert (multiple_param_regressor !=
                    param_regressor), "Either a single regressor or multiple regressors"
            self.single_regressor = False
            assert (isinstance(dispData, list) and isinstance(
                fluidPrevData, list)), warning_text
            assert (len(dispData) == self._p and len(
                fluidPrevData) == self._p), warning_text
        if param_decoder or param_encoder or param_regressor:
            self.param_decoder = param_decoder
            self.param_encoder = param_decoder
            self.param_regressor = param_regressor
            if param_encoder:
                assert (isinstance(dispData, list)), warning_text
                assert (len(dispData) == self._p and self._p ==
                        self._p), warning_text
            if param_decoder:
                assert (isinstance(fluidData, list) and isinstance(
                    fluidPrevData, list)), warning_text
                assert (len(fluidData) == self._p and len(
                    fluidPrevData) == self._p), warning_text

        # print(" ----- Displacement Reduction -----")
        if solidReduc is not None:
            self.reducDisp = None
            reducDisp = solidReduc
            self._disp_latent_dim = reducDisp.latent_dim
        else:
            if isinstance(dispData, list):
                dispData_ = np.hstack(dispData)
            else:
                dispData_ = dispData
            self.reducDisp = PodReducer(latent_dim=rank_disp)
            self.reducDisp.train(dispData_, normalize=norm[1], normalization=normalization[1],
                                 center=center[1], to_copy=False, alg="svd",)
            reducDisp = self.reducDisp

        # print(" ----- Load Reduction -----")

        mean_ = 0
        # One Mean
        for i in range(len(fluidData)):
            mean_ += np.mean(fluidData[i], axis=1)
        mean_ /= len(fluidData)
        # mean_ = 0. * mean_

        # One std
        # std_ = np.std(np.hstack((fluidData)), axis = 1)
        std_ = np.max(np.abs(np.hstack((fluidData))), axis=1)
        # std_ = np.linalg.norm(np.hstack((fluidData)), axis = 1)
        # std_ = np.linalg.norm(np.hstack((fluidData)), axis = 1)/np.linalg.norm(np.hstack((fluidData)), axis = 1)

        reducLoadLocals = []
        self.reducedPrevLoadData = []
        self.reducedLoadData = []
        self.reducedDispData = []
        for i in range(len(fluidData)):
            reducLoadLocal_ = PodReducer(latent_dim=rank_pres)
            reducLoadLocal_.train(fluidData[i], precomp_mean=mean_, normalization=normalization[0], normalize=norm[0],
                                  precomp_std=std_, to_copy=False, alg=alg,)
            reducLoadLocals.append(reducLoadLocal_)
            self.reducedPrevLoadData.append(
                reducLoadLocal_.encode(fluidPrevData[i]))
            self.reducedLoadData.append(reducLoadLocal_.encode(fluidData[i]))
            self.reducedDispData.append(reducDisp.encode(dispData[i]))

        self.reducLoad = ManifInterpReducer(reducLoadLocal_.latent_dim)
        self.reducLoad.train(reducLoadLocals, params, precomp_mean=mean_,
                             precomp_std=std_)
        self.reducLoadLocals = reducLoadLocals

        # print(" ----- Regression -----")
        self.regressor = []
        self._k = params.shape[0]
        self.params = params.copy()
        self.hidden_layers = hidden_layers
        self.kernel = kernel
        self.smoothing = smoothing
        self.degree = degree
        self.epsilon = epsilon
        self.weights = weights
        self.norm_regr = norm_regr
        self.initAlpha = initAlpha

        self.train_regressions(self.kernel, initAlpha, self.degree,
                               self.epsilon, None, None, self.hidden_layers)

        if cleanup:
            self.reducedLoadData = None
            self.reducedDispData = None
            self.reducedPrevLoadData = None

        if False:
            scaleTree = MinMaxScaler()
            scaleTree.fit(params.T)
            self.scaleTree = scaleTree
            self.param_tree = KDTree(scaleTree.transform(params.T))

    def augmentData(self,
                    newdispData,
                    newfluidPrevData,
                    newfluidData,
                    current_t=-1,
                    params=None,
                    solidReduc: RomDimensionalityReducer = None,
                    stepsize=None,
                    computeAngle=False):
        self.sendSignalBasis = None

        if not self._predictedBasis:
            self.initialize_predictions(params, njobs=self.njobs_online, alg=self.alg_square_matrices)

        assert (
            solidReduc is not None or self.reducDisp is not None), "A displacement Encoder is not available"
        if solidReduc is not None:
            dispCoeff = solidReduc.encode(newdispData, high_dim=False)
        else:
            dispCoeff = self.reducDisp.encode(newdispData, high_dim=False)
        prevLoadCoeff = self.reducLoad.encode(newfluidPrevData)
        outLoadCoeff = self.reducLoad.encode(newfluidData)
        input_ = np.vstack((dispCoeff, prevLoadCoeff))

        self.trainIn.appendleft(input_.ravel())
        self.trainOut.appendleft(outLoadCoeff.ravel())

        self.countAugment += 1
        if self.countAugment > self.reTrainThres:
            self._reTrain(self.weights)
            self.retrain_count += 1
            self.retrain_times.append(current_t)
            self.countAugment = 0
            with open("./coSimData/tracked_retraining_instants.npy", 'wb') as f:
                np.save(f, np.array(self.retrain_times))

        if self.updateBasis:
            self.countUpdate += 1
            # self.cloneBasis, self.L = _compute_past(
            #     self.cloneBasis, self.L, self.reducLoad.rom.normalize(
            #         self.reducLoad.rom.center(newfluidData)),
            #     0.99)
            if computeAngle:
                tmpBasis = self.cloneBasis.copy()
            rank1_update(self.cloneBasis, self.reducLoad.rom.normalize(
                self.reducLoad.rom.center(newfluidData)), stepsize)
            # indicator = np.linalg.norm(self.cloneBasis.T @ self.cloneBasis - np.eye(self.reducLoad.latent_dim))
            if computeAngle:
                self.recursive_angles.append(
                    subspace_angles(tmpBasis, self.cloneBasis))

            if self.countUpdate > self.updateThres:
                if solidReduc is not None:
                    self._updateLoadBasis(solidReduc)
                else:
                    self._updateLoadBasis(self.reducDisp)
                self.countUpdate = 0
                self.countAugment = 0

    def _reTrain(self, weights=True):
        print("=== - Retraining the Interpolator - ===")
        t0 = time.time()

        if weights:
            n = len(self.trainIn)
            weights_ = self.sigmoid(np.arange(0, n), n)[::-1]
        else:
            weights_ = None
        if self.new_regressor is None:
            self.new_regressor = self._declareRegressor(
                self.kernel, self.epsilon, self.smoothing, self.degree, self.norm_regr, self.hidden_layers)
        self.new_regressor.train(np.column_stack(self.trainIn),
                                 np.column_stack(self.trainOut), weights=weights_)
        if self.updateOmega:
            self.omega0 += 0.3*(1-self.omega0)
            self._omega_terms = (1-self.omega0, self.omega0)

        t1 = time.time()
        self.retrainingTime.append(t1 - t0)
        with open("./coSimData/tracked_retraining_time.npy", 'wb') as f:
            np.save(f, np.array(self.retrainingTime))

    def _updateLoadBasis(self, solidReduc: PodReducer):
        print("=== - Updating the load basis - ===")
        self.reducLoadLocals.append(copy.deepcopy(self.reducLoad))
        self._p += 1
        self.omega0 = 0.5
        self._omega_terms = (1-self.omega0, self.omega0)

        calibs = self.reducLoad.predictNewModes(
            None, self.reducLoadLocals, self.cloneBasis.copy(), njobs=self.njobs_online, compute_calibration=True)

        self.regressor.append(copy.deepcopy(self.new_regressor))
        self.new_regressor = None

        self._compute_calibrations(njobs=self.njobs_online, alg=self.alg_square_matrices, already_computed_calibs=calibs)

        # for i in range(len(self.trainIn)):
        #     self.trainIn[i] = np.vstack((self.trainIn[i]
        #                                 [:solidReduc.latent_dim, [0]], self.calibrationQs[-1] @  self.trainIn[i][solidReduc.latent_dim:, [0]]))
        # Below, we suppose that the predicted basis (Grassmann-interpolated) is the closest to the current basis (rank1 updated),
        # so no moves were inverted in the former. Hence, we only need to apply the calibration, and no additional treatment is needed.
        tmpIn = np.column_stack(self.trainIn)
        tmpIn[solidReduc.latent_dim:,
              :] = self.calibrationQs[-1] @ tmpIn[solidReduc.latent_dim:, :]
        self.trainIn = collections.deque(tmpIn.T)
        self.trainOut = collections.deque(
            (self.calibrationQs[-1] @ np.column_stack(self.trainOut)).T)

        self.sendSignalBasis = self.calibrationQs[-1]

    def _compute_calibrations(self, njobs=1, alg=None, already_computed_calibs=None):
        compute_calibs = True
        if already_computed_calibs is not None:
            compute_calibs = False
            if already_computed_calibs[0] is None:
                compute_calibs = True
        if compute_calibs:
            alg = _determine_pod_alg_square_matrices(alg=alg, size_=self.reducLoad.pod.modes.shape[1])
            def _core_calibration_computation(currentReducLoadBasis, localReducLoad):
                prod = POD()
                u, _, vh = prod.decompose(
                    currentReducLoadBasis.T @ localReducLoad.pod.modes, thin=False, alg=alg)
                return u @ vh

            self.calibrationQs = Parallel(n_jobs=njobs, backend='loky', prefer='threads')(
                delayed(_core_calibration_computation)(self.reducLoad.pod.modes, local)
                for local in self.reducLoadLocals
            )
        else:
            self.calibrationQs=already_computed_calibs
        self.stacked_calib = np.stack(self.calibrationQs)

    def train_regressions(self, kernel, smoothing, degree, epsilon, weights, norm_regr, hidden_layers=np.array([
            40, 40, 40])):

        for i in range(self._p):
            regressor = self._declareRegressor(
                kernel, epsilon, smoothing, degree, norm_regr, hidden_layers)

            input_ = np.vstack(
                (self.reducedDispData[i], self.reducedPrevLoadData[i]))
            output_ = self.reducedLoadData[i]

            if weights:
                n = input_.shape[1]
                weights_ = self.sigmoid(np.arange(0, n), n)[::-1]
            else:
                weights_ = None

            regressor.train(input_, output_, weights_)

            self.regressor.append(copy.deepcopy(regressor))

    def _declareRegressor(self, kernel, epsilon, smoothing, degree, norm_regr, hidden_layers):
        if self.kernel == "poly":
            regressor = PolynomialDynamicalRegressor(
                smoothing, degree, self.reducLoad.latent_dim)
        elif self.kernel == "polyC":
            regressor = PolynomialRegressor(
                smoothing, degree, True, norm=norm_regr)
        elif self.kernel == "mlp":
            regressor = SKNNRegressor(
                hidden_layers=hidden_layers, norm=norm_regr)
        elif self.kernel == "lasso":
            regressor = PolynomialLassoRegressor(
                degree, criterion='bic', intercept_=True, norm_regr=norm_regr,
                regul_alpha="auto")
        else:
            regressor = RBFRegressor(
                kernel, epsilon, smoothing, degree, norm=norm_regr)

        return regressor

    def invert_saved_data(self):
        for i in range(self._p):
            self.reducedLoadData[i][self.reducLoadLocals[i].pod.invertedModes, :] *= -1
            self.reducedPrevLoadData[i][self.reducLoadLocals[i].pod.invertedModes, :] *= -1

    def initialize_predictions(self, params=None, njobs=1, alg=None):

        if not self._predictedBasis:
            calibs = self.reducLoad.predictNewModes(params, self.reducLoadLocals, njobs=njobs, compute_calibration=True)
            if self.updateBasis:
                self.cloneBasis = self.reducLoad.pod.modes.copy()
                # self.L = (self.cloneBasis.T @ prevLoad) @ (self.cloneBasis.T @ prevLoad).T

            self._compute_calibrations(njobs = njobs, alg=alg, already_computed_calibs=calibs)

            self._predictedBasis = True

    def predict(self, newDisp, newPrevLoad, solidReduc: RomDimensionalityReducer = None, params=None, takes_low_dimensional_disp=False, optimized=True):
        assert (
            solidReduc is not None or self.reducDisp is not None), "A displacement Encoder is not available"

        self.initialize_predictions(params, njobs=self.njobs_online, alg=self.alg_square_matrices)

        if takes_low_dimensional_disp:
            coeffDisp = newDisp
        else:
            if solidReduc is not None:
                coeffDisp = solidReduc.encode(newDisp, high_dim=False)
            else:
                coeffDisp = self.reducDisp.encode(newDisp, high_dim=False)

        if optimized and newDisp.shape[1] == 1:
            new_preds = np.empty(
                (self._p, self.reducLoad.latent_dim))
            for i in range(self._p):
                xTest = np.vstack(
                    (coeffDisp, self.reducLoadLocals[i].encode(newPrevLoad, invertModesAccumulated=True)))
                new_preds[i, :] = self.regressor[i].predict(xTest).ravel()
                new_preds[i, self.reducLoadLocals[i].pod.invertedModesAccumulated] *= -1
            new_preds = np.einsum('ijk,ik->ij', self.stacked_calib, new_preds)
            predicted_ = self.reducLoad.decode(np.dot(new_preds.T, self.reducLoad.weights).reshape((-1, 1)))
        else:
            new_preds = np.empty(
                (self._p, self.reducLoad.latent_dim, newDisp.shape[1]))
            for i in range(self._p):
                xTest = np.vstack(
                    (coeffDisp, self.reducLoadLocals[i].encode(newPrevLoad, invertModesAccumulated=True)))
                new_preds[i, :, :] = self.regressor[i].predict(xTest)
                new_preds[i, self.reducLoadLocals[i].pod.invertedModesAccumulated, :] *= -1
            new_preds = np.einsum('ijk,ikl->ijl', self.stacked_calib, new_preds)
            predicted_ = self.reducLoad.decode(
                np.dot(new_preds.T, self.reducLoad.weights).T)

        if self.new_regressor is not None:
            omega1, omega2 = self._omega_terms

            xTest = np.vstack(
                (coeffDisp, self.reducLoad.encode(newPrevLoad)))
            new_predicted_ = self.reducLoad.decode(
                self.new_regressor.predict(xTest))
            predicted_ *= omega1
            predicted_ += omega2*new_predicted_

        return predicted_

    def save(self, file_name):
        with open(file_name+'.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    # TODO
    # In next version
    #   @property
    # def load_latent_dim(self):

    #     if self._load_latent_dim is None:
    #         try:
    #             self._load_latent_dim = self.reducLoad.latent_dim
    #         except AttributeError:
    #             raise AttributeError(
    #                 "The load dimensionality reducer is not yet constructed. The fluid ROM should be trained")
    #     return self._load_latent_dim

    # @property
    # def disp_latent_dim(self):

    #     if self._disp_latent_dim is None:
    #         try:
    #             self._disp_latent_dim = self.reducDisp.latent_dim
    #         except AttributeError:
    #             raise AttributeError(
    #                 "The displacement dimensionality reducer is not yet constructed. The fluid ROM should be trained")
    #     return self._disp_latent_dim
