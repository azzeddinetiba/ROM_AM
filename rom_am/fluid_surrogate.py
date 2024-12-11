import numpy as np
import collections
from rom_am.regressors.polynomialRegressor import PolynomialRegressor
from rom_am.dimreducers.rom_DimensionalityReducer import RomDimensionalityReducer
from rom_am.dimreducers.rom_am.podReducer import PodReducer
from rom_am.regressors.rbfRegressor import RBFRegressor
from rom_am.regressors.polynomialDynamicalRegressor import PolynomialDynamicalRegressor
from warnings import warn
import pickle


class FluidSurrog:

    def __init__(self, maxLen=6900, reTrainThres=240):
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

    def sigmoid(self, x, n=1):
        s = int(n/2)
        a = 0.1*n
        e =  np.exp(-(x-s)/a)
        m = (1. + e)
        return 1/m

    def train(self, dispData, fluidPrevData, fluidData, input_u=None, kernel='thin_plate_spline', smoothing=9.5e-2,
              rank_pres=.9999, rank_disp=.9999, degree=2, solidReduc: RomDimensionalityReducer = None, epsilon=1.,
              norm=[True, True], center=[True, True], norm_regr="max", params=None, weights=None):
        print(" ----- Load Reduction -----")
        self.reducLoad = PodReducer(latent_dim=rank_pres)
        self.reducLoad.train(fluidData, normalize=norm[0],
                             center=center[0], to_copy=False, alg="svd",)

        print(" ----- Displacement Reduction -----")
        if solidReduc is not None:
            self.reducDisp = None
            reducDisp = solidReduc
            self._disp_latent_dim = reducDisp.latent_dim
        else:
            self.reducDisp = PodReducer(latent_dim=rank_disp)
            self.reducDisp.train(dispData, normalize=norm[1],
                                 center=center[1], to_copy=False, alg="svd",)
            reducDisp = self.reducDisp

        print(" ----- Regression -----")
        input_ = np.vstack((reducDisp.encode(dispData, high_dim=False),
                           self.reducLoad.encode(fluidPrevData)))
        if params is not None:
            input_ = np.vstack((input_, params))
        if input_u is not None:
            input_ = np.vstack((input_, input_u))

        # Store for later updates
        for i in range(input_.shape[1]):
            self.trainIn.appendleft(input_[:, [i]])
            self.trainOut.appendleft(self.reducLoad.reduced_data[:, [i]])

        if kernel == "poly":
            self.regressor = PolynomialDynamicalRegressor(
                smoothing, degree, self.reducLoad.latent_dim)
        elif kernel == "polyC":
            self.regressor = PolynomialRegressor(smoothing, degree, True, norm = norm_regr)
        else:
            self.regressor = RBFRegressor(kernel, epsilon, smoothing, degree, norm = norm_regr)
        self.regressor.train(input_, self.reducLoad.reduced_data, weights)

    def augmentData(self, newdispData, newfluidPrevData, newfluidData, current_t=-1, params = None, solidReduc: RomDimensionalityReducer = None):
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

        self.trainIn.appendleft(input_.copy())
        self.trainOut.appendleft(outLoadCoeff.copy())

        self.countAugment += 1
        if self.countAugment > self.reTrainThres:
            self._reTrain(self.weights)
            self.retrain_count += 1
            self.retrain_times.append(current_t)
            self.countAugment = 0

    def _reTrain(self, weights=True):
        print("=== - Retraining the Interpolator - ===")

        if weights:
            n = len(self.trainIn)
            weights = self.sigmoid(np.arange(0, n), n)[::-1]
        else:
            weights = None
        self.regressor.train(np.hstack(self.trainIn), np.hstack(self.trainOut), weights=weights)

    def predict(self, newDisp, newPrevLoad, input_u=None, solidReduc: RomDimensionalityReducer = None, params=None):
        assert (
            solidReduc is not None or self.reducDisp is not None), "A displacement Encoder is not available"
        if solidReduc is not None:
            coeffDisp = solidReduc.encode(newDisp, high_dim=False)
        else:
            coeffDisp = self.reducDisp.encode(newDisp, high_dim=False)
        xTest = np.vstack((coeffDisp, self.reducLoad.encode(newPrevLoad)))

        if params is not None:
            xTest = np.vstack((xTest, params))

        if input_u is not None:
            xTest = np.vstack((xTest, input_u))

        LoadReconsCoeff = self.regressor.predict(xTest)
        predicted_ = self.reducLoad.decode(LoadReconsCoeff)

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
