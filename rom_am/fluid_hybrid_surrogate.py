import numpy as np
from rom_am.pod import POD
from rom_am.rom import ROM
from scipy.interpolate import RBFInterpolator
import collections
from sklearn.linear_model import Ridge, LassoLarsIC, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from copy import deepcopy


class HybridFluidSurrog:

    def __init__(self, maxLen=6900, reTrainThres=240):
        self.trainIn = collections.deque(maxlen=maxLen)
        self.trainOut = collections.deque(maxlen=maxLen)
        self.maxLen = maxLen
        self.countAugment = 0
        self.reTrainThres = reTrainThres
        self.retrain_count = 0
        self.retrain_times = []

    def train(self, dispData, fluidPrevData, fluidData, input_u=None, kernel='thin_plate_pline', smoothing=9.5e-2,
              rank_pres=.9999, rank_disp=.9999, degree=2, neighbors=None):
        print(" ----- Load Reduction -----")
        podLoad = POD()
        self.romLoad = ROM(podLoad)
        self.romLoad.decompose(
            fluidData, rank=rank_pres, normalize=True, center=True, to_copy=False, alg="svd")
        self.podLoad = podLoad
        output_ = podLoad.pod_coeff.copy().T
        self.scalerOut = MinMaxScaler()
        self.scalerOut.fit(output_)
        # output_ = self.scalerOut.transform(output_)

        print(" ----- Disp Reduction -----")
        podDispLoad = POD()
        self.romDispLoad = ROM(podDispLoad)
        self.romDispLoad.decompose(np.vstack((fluidPrevData, dispData)),
                                   rank=rank_disp, normalize=True,
                                   center=True, to_copy=False, alg="svd")
        self.podDispLoad = podDispLoad
        input_ = podDispLoad.pod_coeff.copy().T
        self.scalerIn = MinMaxScaler()
        self.scalerIn.fit(input_)
        # input_ = self.scalerIn.transform(input_)

        if input_u is not None:
            input_ = np.hstack((input_, input_u.T))

        for i in range(input_.shape[0]):
            self.trainIn.appendleft(input_.T[:, [i]])
            self.trainOut.appendleft(output_.T[:, [i]])

        print(" ----- Regression -----")
        self.kernel = kernel
        self.smoothing = smoothing
        if kernel == "lasso":
            self.func = make_pipeline(
                PolynomialFeatures(degree, include_bias=True),
                MultiOutputRegressor(LassoLarsIC(criterion='bic')))
            self.func.fit(input_.copy(), output_.copy())

            self.nonzeroIds = []
            for i in range(self.podLoad.kept_rank):
                self.nonzeroIds.append(np.argwhere(np.abs(
                    self.func["multioutputregressor"].estimators_[i].coef_) > 1e-11)[:, 0])
        elif kernel == "poly":
            self.func = make_pipeline(
                PolynomialFeatures(degree, include_bias=True),
                Ridge(alpha=smoothing))
            self.func.fit(input_.copy(), output_.copy())
        else:
            self.func = RBFInterpolator(input_.copy(), output_.copy(
            ), kernel=kernel, smoothing=smoothing, neighbors=neighbors)

    def augmentData(self, newdispData, newfluidPrevData, newfluidData, current_t=-1):
        dispCoeff = self.podDispLoad.project(self.romDispLoad.normalize(
            self.romDispLoad.center(np.vstack((newfluidPrevData, newdispData)))))
        outLoadCoeff = self.podLoad.project(
            self.romLoad.normalize(self.romLoad.center(newfluidData)))
        input_ = dispCoeff

        self.trainIn.appendleft(input_.copy())
        self.trainOut.appendleft(outLoadCoeff.copy())

        self.countAugment += 1
        if self.countAugment > self.reTrainThres:
            self._reTrain()
            self.retrain_count += 1
            self.retrain_times.append(current_t)
            self.countAugment = 0

    def _reTrain(self, ):
        print("=== - Retraining the Interpolator - ===")
        if self.kernel == "lasso":
            if not isinstance(self.func, list):
                self.polyF = deepcopy(self.func['polynomialfeatures'])
            self.func = []

            for i in range(len(self.nonzeroIds)):
                func_ = LinearRegression(fit_intercept=True)
                in_retrain = self.polyF.transform(np.hstack(self.trainIn).T)
                in_retrain = in_retrain[:, self.nonzeroIds[i]]
                out_retrain = np.hstack(self.trainOut)[[i], :].T
                func_.fit(in_retrain, out_retrain)
                self.func.append(deepcopy(func_))
            del func_
        elif self.kernel == "poly":
            self.func.fit(np.hstack(self.trainIn).T,
                          np.hstack(self.trainOut).T)
        else:
            self.func = RBFInterpolator(np.hstack(self.trainIn).T,
                                        np.hstack(self.trainOut).T,
                                        kernel=self.kernel, smoothing=self.smoothing)

    def predict(self, newDisp, newPrevLoad, input_u=None):

        inp_ = self.romDispLoad.normalize(
            self.romDispLoad.center(np.vstack((newPrevLoad, newDisp))))
        xTest = self.podDispLoad.project(inp_)

        if input_u is not None:
            xTest = np.vstack((xTest, input_u))
        if self.kernel == "lasso":
            if isinstance(self.func, list):
                LoadReconsCoeff = []
                xTest = self.polyF.transform(xTest.T)
                for i in range(len(self.func)):
                    LoadReconsCoeff.append(self.func[i].predict(
                        xTest[:, self.nonzeroIds[i]]))
                LoadReconsCoeff = np.hstack(LoadReconsCoeff).T
            else:
                LoadReconsCoeff = self.func.predict(xTest.T).T
        elif self.kernel == "poly":
            LoadReconsCoeff = self.func.predict(xTest.T).T
        else:
            LoadReconsCoeff = self.func(xTest.T).T

        # LoadReconsCoeff = self.scalerOut.transform(LoadReconsCoeff.T).T

        predicted_ = self.romLoad.decenter(self.romLoad.denormalize(
            self.podLoad.inverse_project(LoadReconsCoeff)))
        return predicted_
