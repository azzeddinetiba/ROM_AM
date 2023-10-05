from sklearn.linear_model import LassoLarsIC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from rom_am.regressors.rom_regressor import *


class PolynomialLassoRegressor(RomRegressor):

    def __init__(self, poly_degree, criterion='bic', intercept_=True) -> None:
        super().__init__()
        self.criterion = criterion
        self.poly_degree = poly_degree
        self.intercept_ = intercept_

    def train(self, input_data, output_data):
        super().train(input_data, output_data)

        self.regr_model = make_pipeline(
            PolynomialFeatures(self.poly_degree, include_bias=self.intercept_), MultiOutputRegressor(LassoLarsIC(criterion=self.criterion),))
        self.regr_model.fit(input_data.T, output_data.T)

        self.nonzeroIds = []
        for i in range(self.output_dim):
            self.nonzeroIds.append(np.argwhere(np.abs(
                self.regr_model["multioutputregressor"].estimators_[i].coef_) > 1e-9)[:, 0])

    def predict(self, new_input):

        # Instead of self.regr_model.predict(new_input.T).T, the following is faster :
        self.polyFeatures = self.regr_model["polynomialfeatures"].transform(
            new_input.T)

        def mult_(proc):
            linear_ = self.polyFeatures[:, self.nonzeroIds[proc]] @ self.regr_model["multioutputregressor"].estimators_[
                proc].coef_[self.nonzeroIds[proc]].reshape((-1, 1))
            return linear_ + self.regr_model["multioutputregressor"].estimators_[proc].intercept_

        res = np.empty((self.output_dim, new_input.shape[1]))
        for i in range(self.output_dim):
            res[i, :] = mult_(i).ravel()

        # TODO is this even faster ?
        # from joblib import Parallel, delayed
        # n_CPUs = 8
        # res1 = Parallel(n_jobs=n_CPUs)(delayed(mult_)(i)
        #                               for i in range(self.output_dim))
        # res = np.hstack((res1))

        return res
