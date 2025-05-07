from sklearn.linear_model import LassoLarsIC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from rom_am.regressors.rom_regressor import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import Ridge


class PolynomialLassoRegressor(RomRegressor):

    def __init__(self, poly_degree, criterion='bic', intercept_=True, norm_regr = "max", regul_alpha="auto") -> None:
        super().__init__()
        self.criterion = criterion
        self.poly_degree = poly_degree
        self.intercept_ = intercept_
        self.regul_alpha = regul_alpha
        self.norm = norm_regr
        self.updated_regr_model = None

    def train(self, input_data, output_data, weights):
        super().train(input_data, output_data)

        if self.norm is None:
            scale = None
        elif self.norm == "max":
            scale = MaxAbsScaler()
        elif self.norm == "std":
            scale = StandardScaler()
        self.scale = scale

        self.regr_model = make_pipeline(scale,
            PolynomialFeatures(self.poly_degree, include_bias=self.intercept_),
            MultiOutputRegressor(LassoLarsIC(criterion=self.criterion),))
        self.regr_model.fit(input_data.T, output_data.T)

        self.nonzeroIds = []
        for i in range(self.output_dim):
            self.nonzeroIds.append(np.argwhere(np.abs(
                self.regr_model["multioutputregressor"].estimators_[i].coef_) > 1e-9)[:, 0])

    def predict(self, new_input):

        if self.updated_regr_model is None:
            # Instead of self.regr_model.predict(new_input.T).T, the following is faster :
            polyFeatures = self.regr_model["polynomialfeatures"].transform(
                new_input.T)

            def mult_(proc, polyFeatures_):
                linear_ = polyFeatures_[:, self.nonzeroIds[proc]] @ self.regr_model["multioutputregressor"].estimators_[
                    proc].coef_[self.nonzeroIds[proc]].reshape((-1, 1))
                return linear_ + self.regr_model["multioutputregressor"].estimators_[proc].intercept_

            res = np.empty((self.output_dim, new_input.shape[1]))
            for i in range(self.output_dim):
                res[i, :] = mult_(i, polyFeatures).ravel()
        else:
            res = self.updated_regr_model.predict(new_input[self.nonzeroIds, :])

        # TODO is this even faster ?
        # from joblib import Parallel, delayed
        # n_CPUs = 8
        # res1 = Parallel(n_jobs=n_CPUs)(delayed(mult_)(i)
        #                               for i in range(self.output_dim))
        # res = np.hstack((res1))

        return res

    def update(self, input_data, output_data, weights):

        if self.regul_alpha == "auto":
            regul_alpha = self.compute_coefficient(input_data.shape[1])
        else:
            regul_alpha = self.regul_alpha

        self.updated_regr_model = make_pipeline(self.scale,
            PolynomialFeatures(self.poly_degree), Ridge(alpha=regul_alpha))
        self.updated_regr_model.train(input_data[self.nonzeroIds, :].T, output_data.T, weights)


    def compute_coefficient(self, x):
        """
        Compute a coefficient that transitions smoothly from ~1e-1 (at x=0) to ~1e-8 (at x=250).

        Args:
            x (float or array-like): Number of points (0 to ~250).

        Returns:
            float or array: Corresponding coefficient.
        """
        # Sigmoid parameters
        x0 = 125  # Midpoint of transition
        k = 0.05  # Steepness of transition

        # Compute sigmoid
        sigmoid = 1 / (1 + np.exp(-k * (x - x0)))

        # Compute coefficient in log-space
        exponent = -1 - 7 * sigmoid
        y = 10 ** exponent

        return y
