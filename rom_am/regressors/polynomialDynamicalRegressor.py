from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from rom_am.regressors.rom_regressor import *


class PolynomialDynamicalRegressor(RomRegressor):

    def __init__(self, regul_alpha, poly_degree, stateDim) -> None:
        super().__init__()
        self.regul_alpha = regul_alpha
        self.poly_degree = poly_degree
        self.stateDim = stateDim

    def train(self, input_data, output_data):
        super().train(input_data, output_data)

        self.regr_model = make_pipeline(ColumnTransformer(transformers=[
                ('polynomialCoefficients', PolynomialFeatures(
                    self.degree, include_bias=True), slice(0, self.stateDim)),
            ],
                remainder='passthrough'), MinMaxScaler(),
                Ridge(alpha=self.regul_alpha))
        self.regr_model.fit(input_data.T, output_data.T)

    def predict(self, new_input):
        return self.regr_model.predict(new_input.T).T
