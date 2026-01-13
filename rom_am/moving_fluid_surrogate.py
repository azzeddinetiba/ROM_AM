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
from rom_am.utils import rank1_update
from rom_am.rpod import _compute_past

class MovingFluidSurrog:

    def __init__(self, maxLen=6900, reTrainThres=240, parametric=False, updateBasis=False, updateThres=300):
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
        self.omega0 = 0.4
        self.updateBasis = updateBasis
        self.updateThres = updateThres
        self.sendSignalBasis = None

    def sigmoid(self, x, n=1):
        s = int(n/2)
        a = 0.1*n
        e =  np.exp(-(x-s)/a)
        m = (1. + e)
        return 1/m

    def train(self, dispData, fluidPrevData, fluidData, kernel='thin_plate_spline', smoothing=9.5e-2,
              rank_pres=.9999, rank_disp=.9999, degree=2, solidReduc: RomDimensionalityReducer = None, epsilon=1.,
              norm=[True, True], center=[True, True], norm_regr="max", params=None, weights=None, param_encoder=False,
              param_decoder=False, param_regressor=False, multiple_param_regressor=False, hidden_layers=np.array([
            40, 40, 40]), normalization=["norm", "norm"]):

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
        std_ = np.max(np.abs(np.hstack((fluidData))), axis = 1)
        # std_ = np.linalg.norm(np.hstack((fluidData)), axis = 1)
        # std_ = np.linalg.norm(np.hstack((fluidData)), axis = 1)/np.linalg.norm(np.hstack((fluidData)), axis = 1)


        reducLoadLocals = []
        self.reducedPrevLoadData = []
        self.reducedLoadData = []
        self.reducedDispData = []
        for i in range(len(fluidData)):
            reducLoadLocal_ = PodReducer(latent_dim=rank_pres)
            reducLoadLocal_.train(fluidData[i], precomp_mean = mean_, normalization=normalization[0], normalize=norm[0],
                                precomp_std=std_, to_copy=False, alg="svd",)
            reducLoadLocals.append(reducLoadLocal_)
            self.reducedPrevLoadData.append(reducLoadLocal_.encode(fluidPrevData[i]))
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
        # Keep for later

        if False:
            scaleTree = MinMaxScaler()
            scaleTree.fit(params.T)
            self.scaleTree = scaleTree
            self.param_tree = KDTree(scaleTree.transform(params.T))



    def augmentData(self, newdispData, newfluidPrevData, newfluidData, current_t=-1, params = None, solidReduc: RomDimensionalityReducer = None, stepsize=0.2):
        self.sendSignalBasis = None
        assert (
            solidReduc is not None or self.reducDisp is not None), "A displacement Encoder is not available"
        if solidReduc is not None:
            dispCoeff = solidReduc.encode(newdispData, high_dim=False)
        else:
            dispCoeff = self.reducDisp.encode(newdispData, high_dim=False)
        prevLoadCoeff = self.reducLoad.encode(newfluidPrevData)
        outLoadCoeff = self.reducLoad.encode(newfluidData)
        input_ = np.vstack((dispCoeff, prevLoadCoeff))

        self.trainIn.appendleft(input_.copy())
        self.trainOut.appendleft(outLoadCoeff.copy())

        self.countAugment += 1
        if self.countAugment > self.reTrainThres:
            self._reTrain(self.weights)
            self.retrain_count += 1
            self.retrain_times.append(current_t)
            self.countAugment = 0

        if self.updateBasis:
            self.countUpdate += 1
            # self.cloneBasis, self.L = _compute_past(
            #     self.cloneBasis, self.L, self.reducLoad.rom.normalize(
            #         self.reducLoad.rom.center(newfluidData)),
            #     0.99)
            rank1_update(self.cloneBasis, self.reducLoad.rom.normalize(
                self.reducLoad.rom.center(newfluidData)), stepsize)
            # indicator = np.linalg.norm(self.cloneBasis.T @ self.cloneBasis - np.eye(self.reducLoad.latent_dim))

            if self.countUpdate > self.updateThres:
                self._updateLoadBasis(solidReduc)
                self.countUpdate = 0
                self.countAugment = 0



    def _reTrain(self, weights=True):
        # pass
        print("=== - Retraining the Interpolator - ===")
        if weights:
            n = len(self.trainIn)
            weights_ = self.sigmoid(np.arange(0, n), n)[::-1]
        else:
            weights_ = None
        if self.new_regressor is None:
            self.new_regressor = self._declareRegressor(
                self.kernel, self.epsilon, self.smoothing, self.degree, self.norm_regr, self.hidden_layers)
        self.new_regressor.train(np.hstack(self.trainIn),
                                np.hstack(self.trainOut), weights=weights_)
        self.omega0 += 0.3*(1-self.omega0)


    # def _reset_predicted_basis(self, ):
    #     self._predictedBasis = False
    #     self.invert_saved_data()

    def _updateLoadBasis(self, solidReduc: PodReducer, override_weights=False):
        print("=== - Updating the load basis - ===")
        self.new_regressor = None
        if override_weights:
            weights_ = None
        else:
            weights_ = self.weights
        self.reducedDispData.append(np.hstack(self.trainIn)[
                                    :solidReduc.latent_dim, :])
        self.reducedPrevLoadData.append(
            np.hstack(self.trainIn)[solidReduc.latent_dim:, :])
        self.reducedLoadData.append(np.hstack(self.trainOut).copy())
        # self.trainIn.clear()
        # self.trainOut.clear()
        self.reducLoadLocals.append(copy.deepcopy(self.reducLoad))
        self._p += 1
        self.omega0 = 0.4
        for i in range(self._p):
            self.reducLoadLocals[i].pod.invertedModes = []
        self.reducLoad.predictNewModes(None, self.reducLoadLocals, self.cloneBasis.copy())


        self.invert_saved_data()
        self.regressor = []
        self.train_regressions(solidReduc, self.kernel, self.smoothing, self.degree,
                               self.epsilon, weights_, self.norm_regr, self.hidden_layers)

        for i in range(len(self.trainIn)):
            self.trainIn[i] = np.vstack((self.trainIn[i]
                                        [:solidReduc.latent_dim, [0]], self.calibrationQs[-1] @  self.trainIn[i][solidReduc.latent_dim:, [0]]))
            self.trainOut[i] = self.calibrationQs[-1] @ self.trainOut[i]

        # for i in range(self.reducedDispData[-1].shape[1]):
        #     self.trainIn.append(np.vstack(
        #         (self.reducedDispData[-1][:, [i]],
        #          self.calibrationQs[-1] @  self.reducedPrevLoadData[-1][:, [i]]
        #          )
        #     )
        #     )
        #     self.trainOut.append(
        #         self.calibrationQs[-1] @ self.reducedLoadData[-1][:, [i]])
        self.sendSignalBasis = self.calibrationQs[-1].copy()


    def train_regressions(self, reducDisp: PodReducer, kernel, smoothing, degree, epsilon, weights, norm_regr, hidden_layers=np.array([
            40, 40, 40])):
        self.calibrationQs = []
        for i in range(self._p):
            prod = POD()
            u, _, vh = prod.decompose(self.reducLoad.pod.modes.T @ self.reducLoadLocals[i].pod.modes, thin=False)
            self.calibrationQs.append(u @ vh)
            # self.calibrationQs.append(self.reducLoad.pod.modes.T @ self.reducLoadLocals[i].pod.modes)
            # trainIn = collections.deque(maxlen=self.maxLen)
            # trainOut = collections.deque(maxlen=self.maxLen)

            regressor = self._declareRegressor(kernel, epsilon, smoothing, degree, norm_regr, hidden_layers)

            if False:
                input_ = np.vstack((self.reducedDispData[i], self.reducedPrevLoadData[i]))
                output_ = self.reducedLoadData[i].copy()
            else:
                input_ = np.vstack((self.reducedDispData[i],
                                self.calibrationQs[i] @ self.reducedPrevLoadData[i]))
                output_ = self.calibrationQs[i] @ self.reducedLoadData[i]
            if weights:
                n = input_.shape[1]
                weights_ = self.sigmoid(np.arange(0, n), n)[::-1]
            else:
                weights_ = None

            regressor.train(input_, output_, weights_)
            # Store for later updates
            # for j in range(input_.shape[1]):
            #     self.trainIn.appendleft(input_[:, [j]].copy())
            #     self.trainOut.appendleft(output_[:, [j]].copy())

            self.regressor.append(copy.deepcopy(regressor))
            # Store for later updates
            # for j in range(input_.shape[1]):
            #     trainIn.appendleft(input_[:, [j]])
            #     trainOut.appendleft(usedReducedLoad.encode(fluidData[i])[:, j])
        # self.trainIn.append(copy.deepcopy(trainIn))
        # self.trainOut.append(copy.deepcopy(trainOut))

    def _declareRegressor(self, kernel, epsilon, smoothing, degree, norm_regr, hidden_layers):
        if self.kernel == "poly":
            regressor = PolynomialDynamicalRegressor(
                smoothing, degree, self.reducLoad.latent_dim)
        elif self.kernel == "polyC":
            regressor = PolynomialRegressor(smoothing, degree, True, norm = norm_regr)
        elif self.kernel == "mlp":
            regressor = SKNNRegressor(hidden_layers=hidden_layers, norm = norm_regr)
        elif self.kernel == "lasso":
            regressor = PolynomialLassoRegressor(
                degree, criterion='bic', intercept_=True, norm_regr=norm_regr, regul_alpha="auto")
        else:
            regressor = RBFRegressor(kernel, epsilon, smoothing, degree, norm = norm_regr)

        return regressor

    def invert_saved_data(self):
        for i in range(self._p):
            self.reducedLoadData[i][self.reducLoadLocals[i].pod.invertedModes, :] *= -1
            self.reducedPrevLoadData[i][self.reducLoadLocals[i].pod.invertedModes, :] *= -1


    def initialize_predictions(self, solidReduc: RomDimensionalityReducer = None, params=None, override_weights=False, cleanup=True, initAlpha=6e-3):

        if not self._predictedBasis:
            self.reducLoad.predictNewModes(params, self.reducLoadLocals)
            if self.updateBasis:
                self.cloneBasis = self.reducLoad.pod.modes.copy()
                # self.L = (self.cloneBasis.T @ prevLoad) @ (self.cloneBasis.T @ prevLoad).T
            self.invert_saved_data()
            if override_weights:
                weights_ = None
            else:
                weights_ = self.weights
            self.train_regressions(solidReduc, self.kernel, initAlpha, self.degree,
                                   self.epsilon, weights_, self.norm_regr, self.hidden_layers)
            self._predictedBasis = True
            if cleanup:
                self.reducedLoadData = None
                self.reducedDispData = None
                self.reducedPrevLoadData = None


    def predict(self, newDisp, newPrevLoad, input_u=None, solidReduc: RomDimensionalityReducer = None, params=None, k=None, nnls_tikhonov=None, cleanup=True, override_weights=True, initAlpha=6e-3):
        assert (
            solidReduc is not None or self.reducDisp is not None), "A displacement Encoder is not available"

        self.initialize_predictions(solidReduc, params, override_weights, cleanup, initAlpha)

        if solidReduc is not None:
            coeffDisp = solidReduc.encode(newDisp, high_dim=False)
        else:
            coeffDisp = self.reducDisp.encode(newDisp, high_dim=False)


        xTest = np.vstack(
            (coeffDisp, self.reducLoad.encode(newPrevLoad)))
        new_preds = np.empty(
            (self._p, self.reducLoad.latent_dim, newDisp.shape[1]))
        for i in range(self._p):
            new_preds[i, :, :] = self.regressor[i].predict(xTest)
        predicted_ = self.reducLoad.decode(np.dot(new_preds.T, self.reducLoad.weights).T)
        if self.new_regressor is not None:
            new_predicted_ = self.reducLoad.decode(self.new_regressor.predict(xTest))
            predicted_ = (1-self.omega0)*predicted_ + self.omega0*new_predicted_

        return predicted_

    def save(self, file_name):
        with open(file_name+'.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    #TODO
    #In next version
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
