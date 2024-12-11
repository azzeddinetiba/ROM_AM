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

        self.regr_model = make_pipeline(scale,
            PolynomialFeatures(self.poly_degree), Ridge(alpha=self.regul_alpha))
        self.regr_model.fit(input_data.T, output_data.T, **{'ridge__sample_weight': weights})

    def predict(self, new_input):
        return self.regr_model.predict(new_input.T).T
