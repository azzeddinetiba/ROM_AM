import numpy as np
from scipy.interpolate import RBFInterpolator
from rom_am.regressors.rom_regressor import *


class RBFRegressor(RomRegressor):

    def __init__(self, kernel="thin_plate_spline", epsilon=1., smoothing=0., degree=1) -> None:
        super().__init__()

        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.smoothing = smoothing

    def train(self, input_data, output_data):

        super().train(input_data, output_data)

        self.regr_model = RBFInterpolator(
            input_data.T, output_data.T, kernel=self.kernel,
            epsilon=self.epsilon, degree=self.degree, smoothing=self.smoothing)

    def predict(self, new_input):
        return self.regr_model(new_input.T).T
