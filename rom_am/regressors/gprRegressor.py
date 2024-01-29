from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import make_pipeline
from rom_am.regressors.rom_regressor import *


class GprRegressor(RomRegressor):

    def __init__(self, return_std = True) -> None:
        super().__init__()
        self.return_std = return_std

    def train(self, input_data, output_data):
        super().train(input_data, output_data)

        kernel = RBF()
        self.regr_model = GaussianProcessRegressor(
            kernel=kernel, ).fit(input_data.T, output_data.T)

    def predict(self, new_input):
        if self.return_std:
            res, std = self.regr_model.predict(new_input.T).T
        else:
            res = self.regr_model.predict(new_input.T).T
        return res
