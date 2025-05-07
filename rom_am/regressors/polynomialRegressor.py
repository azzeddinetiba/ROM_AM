from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from rom_am.regressors.rom_regressor import *


class PolynomialRegressor(RomRegressor):

    def __init__(self, regul_alpha, poly_degree, std=False, norm = None) -> None:
        super().__init__()
        self.regul_alpha = regul_alpha
        self.poly_degree = poly_degree
        self.norm = norm

    def train(self, input_data, output_data, weights):
        super().train(input_data, output_data)
        if self.norm is None:
            scale = None
        elif self.norm == "max":
            scale = MaxAbsScaler()
        elif self.norm == "std":
            scale = StandardScaler()

        if self.regul_alpha == "auto":
            regul_alpha = self.compute_coefficient(input_data.shape[1])
        else:
            regul_alpha = self.regul_alpha
        self.regr_model = make_pipeline(scale,
            PolynomialFeatures(self.poly_degree), Ridge(alpha=regul_alpha))
        self.regr_model.fit(input_data.T, output_data.T, **{'ridge__sample_weight': weights})

    def predict(self, new_input):
        return self.regr_model.predict(new_input.T).T

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

    def update(self, input_data, output_data, weights):
        self.train(input_data.T, output_data.T, weights)
