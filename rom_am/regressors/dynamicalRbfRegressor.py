from sklearn.linear_model import LassoLarsIC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from rom_am.regressors.rbfRegressor import RBFRegressor
import numpy as np

class DynamicalRBFRegressor(RBFRegressor):

    def __init__(self, kernel="thin_plate_spline", epsilon=1., degree=1) -> None:
        super().__init__(kernel=kernel, epsilon=epsilon, degree=degree)

    def train(self, input_data, output_data, previous_input_data):
        super().train(np.vstack((input_data, previous_input_data)), output_data)
        self.input_dim = input_data.shape[0]

    def predict(self, new_input, previous_input, alpha=None):
        res = super().predict(np.vstack((new_input, previous_input)))
        return res
