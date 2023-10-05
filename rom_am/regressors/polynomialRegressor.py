from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from rom_am.regressors.rom_regressor import *


class PolynomialRegressor(RomRegressor):

    def __init__(self, regul_alpha, poly_degree) -> None:
        super().__init__()
        self.regul_alpha = regul_alpha
        self.poly_degree = poly_degree

    def train(self, input_data, output_data):
        super().train(input_data, output_data)

        self.regr_model = make_pipeline(
            PolynomialFeatures(self.poly_degree), Ridge(alpha=self.regul_alpha))
        self.regr_model.fit(input_data.T, output_data.T)

    def predict(self, new_input):
        return self.regr_model.predict(new_input.T).T
