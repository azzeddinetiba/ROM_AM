from sklearn.linear_model import LassoLarsIC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from rom_am.regressors.rom_regressor import *


class PolynomialLassoRegressor(RomRegressor):

    def __init__(self, poly_degree, criterion='bic') -> None:
        super().__init__()
        self.criterion = criterion
        self.poly_degree = poly_degree

    def train(self, input_data, output_data):
        super().train(input_data, output_data)

        self.regr_model = make_pipeline(
            PolynomialFeatures(self.poly_degree), MultiOutputRegressor(LassoLarsIC(criterion=self.criterion)))
        self.regr_model.fit(input_data.T, output_data.T)

    def predict(self, new_input):

        super()._check_predict(new_input)
        return self.regr_model.predict(new_input.T).T
