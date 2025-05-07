import numpy as np
import collections
from rom_am.regressors.polynomialRegressor import PolynomialRegressor
from rom_am.dimreducers.rom_DimensionalityReducer import RomDimensionalityReducer
from rom_am.dimreducers.rom_am.podReducer import PodReducer
from rom_am.dimreducers.rom_am.manifReducer import ManifInterpReducer
from rom_am.regressors.rbfRegressor import RBFRegressor
from rom_am.regressors.polynomialDynamicalRegressor import PolynomialDynamicalRegressor
from warnings import warn
from . import utils
import pickle
import copy
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
import time


class FluidSurrog:

    def __init__(self, maxLen=6900, reTrainThres=240, parametric=False):
        self.trainIn = collections.deque(maxlen=maxLen)
        self.trainOut = collections.deque(maxlen=maxLen)
        self.maxLen = maxLen
        self.countAugment = 0
        self.reTrainThres = reTrainThres
        self.retrain_count = 0
        self.retrain_times = []
        self.reTrainKernel = None
        self.reTrainSmoothing = None
        self._disp_latent_dim = None
        self._load_latent_dim = None
        self.weights = False
        self.param_encoder = None
        self.param_decoder = None
        self.param_regressor = None
        self.multiple_regressor = None
        self.single_regressor = None
        self._p = None
        self._predictedBasis = None
        self.sendSignalBasis = None
        self.retrainingTime = []

    def sigmoid(self, x, n=1):
        s = int(n/2)
        a = 0.1*n
        e =  np.exp(-(x-s)/a)
        m = (1. + e)
        return 1/m

    def train(self, dispData, fluidPrevData, fluidData, input_u=None, kernel='thin_plate_spline', smoothing=9.5e-2,
              rank_pres=.9999, rank_disp=.9999, degree=2, solidReduc: RomDimensionalityReducer = None, epsilon=1.,
              norm=[True, True], center=[True, True], norm_regr="max", params=None, weights=None, param_encoder=False,
              param_decoder=False, param_regressor=False, multiple_param_regressor=False, normalization=["norm", "norm"],
              precomputedReducLoad=None):

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


        print(" ----- Load Reduction -----")
        if precomputedReducLoad is None:
            if self.param_decoder:
                reducLoadLocals = []
                for i in range(len(fluidData)):
                    reducLoadLocal_ = PodReducer(latent_dim=rank_pres)
                    reducLoadLocal_.train(fluidData[i], normalize=norm[0], normalization=normalization[0],
                                        center=center[0], to_copy=False, alg="svd",)
                    reducLoadLocals.append(reducLoadLocal_)
                self.reducLoad = ManifInterpReducer(reducLoadLocal_.latent_dim)
                self.reducLoad.train(reducLoadLocals, params)
                self.reducLoadLocals = reducLoadLocals

            else:
                if isinstance(fluidData, list):
                    fluidData_ = np.hstack(fluidData)
                else:
                    fluidData_ = fluidData
                self.reducLoad = PodReducer(latent_dim=rank_pres)
                self.reducLoad.train(fluidData_, normalize=norm[0],
                                    center=center[0], to_copy=False, alg="svd",)
        else:
            self.reducLoad = precomputedReducLoad

        print(" ----- Displacement Reduction -----")
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

        print(" ----- Regression -----")
        if self.multiple_regressor:
            self.regressor = []
            self.trainIn = []
            self.trainOut = []
            self._k = params.shape[0]
            self.params = params.copy()

            for i in range(self._p):
                if self.param_decoder:
                    usedReducedLoad = reducLoadLocals[i] # Local
                else:
                    usedReducedLoad = self.reducLoad # Global
                trainIn = collections.deque(maxlen=self.maxLen)
                trainOut = collections.deque(maxlen=self.maxLen)
                input_ = np.vstack((reducDisp.encode(dispData[i], high_dim=False),
                                usedReducedLoad.encode(fluidPrevData[i])))
                if kernel == "poly":
                    regressor = PolynomialDynamicalRegressor(
                        smoothing, degree, usedReducedLoad.latent_dim)
                elif kernel == "polyC":
                    regressor = PolynomialRegressor(smoothing, degree, True, norm = norm_regr)
                else:
                    regressor = RBFRegressor(kernel, epsilon, smoothing, degree, norm = norm_regr)
                regressor.train(input_, usedReducedLoad.encode(fluidData[i]), weights)
                self.regressor.append(copy.deepcopy(regressor))
                # Store for later updates
                for j in range(input_.shape[1]):
                    trainIn.appendleft(input_[:, [j]])
                    trainOut.appendleft(usedReducedLoad.encode(fluidData[i])[:, j])
                self.trainIn.append(copy.deepcopy(trainIn))
                self.trainOut.append(copy.deepcopy(trainOut))


            scaleTree = MinMaxScaler()
            scaleTree.fit(params.T)
            self.scaleTree = scaleTree
            self.param_tree = KDTree(scaleTree.transform(params.T))



        if self.single_regressor:
            if params is not None:
                self.param_regressor = True

            input_ = np.vstack((reducDisp.encode(dispData, high_dim=False),
                            self.reducLoad.encode(fluidPrevData)))
            if self.param_regressor:
                input_ = np.vstack((input_, params))
            if input_u is not None:
                input_ = np.vstack((input_, input_u))

            # Store for later updates
            self.trainIn = collections.deque(input_.T, maxlen=self.maxLen)
            # We add this to keep the solutions conformal with the previous version, where a for loop was used to append to the deque
            self.trainIn = collections.deque(
                np.flip(np.column_stack(self.trainIn), axis=1).T, maxlen=self.maxLen)

            self.trainOut = collections.deque(self.reducLoad.reduced_data.T, maxlen=self.maxLen)
            # We add this to keep the solutions conformal with the previous version, where a for loop was used to append to the deque
            self.trainOut = collections.deque(
                np.flip(np.column_stack(self.trainOut), axis=1).T, maxlen=self.maxLen)
            # for i in range(input_.shape[1]):
            #     self.trainIn.appendleft(input_[:, [i]])
            #     self.trainOut.appendleft(self.reducLoad.reduced_data[:, [i]])

            if kernel == "poly":
                self.regressor = PolynomialDynamicalRegressor(
                    smoothing, degree, self.reducLoad.latent_dim)
            elif kernel == "polyC":
                self.regressor = PolynomialRegressor(smoothing, degree, True, norm = norm_regr)
            else:
                self.regressor = RBFRegressor(kernel, epsilon, smoothing, degree, norm = norm_regr)
            self.regressor.train(input_, self.reducLoad.reduced_data, weights)

    def augmentData(self, newdispData, newfluidPrevData, newfluidData, current_t=-1, params = None, solidReduc: RomDimensionalityReducer = None, changeTheBasis=None):
        assert (
            solidReduc is not None or self.reducDisp is not None), "A displacement Encoder is not available"
        if solidReduc is not None:
            dispCoeff = solidReduc.encode(newdispData, high_dim=False)
        else:
            dispCoeff = self.reducDisp.encode(newdispData, high_dim=False)
        prevLoadCoeff = self.reducLoad.encode(newfluidPrevData)
        outLoadCoeff = self.reducLoad.encode(newfluidData)
        input_ = np.vstack((dispCoeff, prevLoadCoeff))
        if params is not None:
            input_ = np.vstack((input_, params))

        self.trainIn.appendleft(input_.ravel())
        self.trainOut.appendleft(outLoadCoeff.ravel())

        if changeTheBasis is not None:

            tmpIn = np.column_stack(self.trainIn)
            tmpIn[:solidReduc.latent_dim, :] = changeTheBasis @ tmpIn[:solidReduc.latent_dim, :]
            self.trainIn = collections.deque(tmpIn.T, maxlen = self.maxLen)
            # for i in range(1, len(self.trainIn)):
            #     self.trainIn[i] = np.vstack((changeTheBasis @ self.trainIn[i][:solidReduc.latent_dim, [
            #                                 0]], self.trainIn[i][solidReduc.latent_dim:, [0]]))


        self.countAugment += 1
        if self.countAugment > self.reTrainThres:
            self._reTrain(self.weights)
            self.retrain_count += 1
            self.retrain_times.append(current_t)
            self.countAugment = 0
            with open("./coSimData/retraining_instants.npy", 'wb') as f:
                np.save(f, np.array(self.retrain_times))

    def _reTrain(self, weights=True):
        print("=== - Retraining the Interpolator - ===")

        t0 = time.time()
        if weights:
            n = len(self.trainIn)
            weights = self.sigmoid(np.arange(0, n), n)[::-1]
        else:
            weights = None
        self.regressor.train(np.column_stack(self.trainIn), np.column_stack(self.trainOut), weights=weights)
        t1 = time.time()
        self.retrainingTime.append(t1 - t0)
        with open("./coSimData/retraining_time.npy", 'wb') as f:
            np.save(f, np.array(self.retrainingTime))

    def predict(self, newDisp, newPrevLoad, input_u=None, solidReduc: RomDimensionalityReducer = None, params=None, k=None, nnls_tikhonov=None, predict_low_dimensional=False):
        assert (
            solidReduc is not None or self.reducDisp is not None), "A displacement Encoder is not available"
        if self.param_decoder:
            if self._predictedBasis is None:
                if not self._predictedBasis:
                    self.reducLoad.predictNewModes(params, self.reducLoadLocals)

        if solidReduc is not None:
            coeffDisp = solidReduc.encode(newDisp, high_dim=False)
        else:
            coeffDisp = self.reducDisp.encode(newDisp, high_dim=False)
        if not (self.param_decoder and self.multiple_regressor):
            xTest = np.vstack((coeffDisp, self.reducLoad.encode(newPrevLoad)))

        if not self.multiple_regressor:
            if params is not None:
                xTest = np.vstack((xTest, params))

            if input_u is not None:
                xTest = np.vstack((xTest, input_u))

            LoadReconsCoeff = self.regressor.predict(xTest)
            if predict_low_dimensional:
                predicted_ = LoadReconsCoeff
            else:
                predicted_ = self.reducLoad.decode(LoadReconsCoeff)
        else:
            if k is None:
                k = self._k + 1
            _, closestParamsIds = self.param_tree.query(
                (self.scaleTree.transform(params.T).T).ravel(), k=k)
            new_preds = np.empty(
                (k, self.reducLoad.latent_dim, newDisp.shape[1]))
            for i in range(len(closestParamsIds)):
                if self.param_decoder:
                    xTest = np.vstack(
                        (coeffDisp, self.reducLoadLocals[i].encode(newPrevLoad, invertModes=True)))
                new_preds[i, :, :] = self.regressor[closestParamsIds[i]].predict(
                    xTest)
                if self.param_decoder:
                    new_preds[i, self.reducLoadLocals[i].pod.invertedModes, :] = - \
                        new_preds[i, self.reducLoadLocals[i].pod.invertedModes, :]
                    new_preds = self.reducLoad.calibrationQ @ new_preds
            weights = utils.cnvx_nnls(self.scaleTree.transform(params.T).T,
                                      self.scaleTree.transform(self.params[:, closestParamsIds].T).T, mu=nnls_tikhonov)

            predicted_ = self.reducLoad.decode(np.dot(new_preds.T, weights).T)

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
