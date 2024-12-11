import numpy as np
from scipy.interpolate import RBFInterpolator
from rom_am.regressors.rom_regressor import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler


class RBFRegressor(RomRegressor):

    def __init__(self, kernel="thin_plate_spline", epsilon=1., smoothing=0., degree=1, norm=None) -> None:
        super().__init__()

        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.smoothing = smoothing
        self.norm = norm
        self.scale = None

    def train(self, input_data, output_data, weights=None):

        super().train(input_data, output_data)

        if self.norm is None:
            trnsf_input_data = input_data.T
        else:
            if self.norm == "max":
                scale = MaxAbsScaler()
            elif self.norm == "std":
                scale = StandardScaler()
            self.scale = scale
            scale.fit(input_data.T)
            trnsf_input_data = scale.transform(input_data.T)

        self.regr_model = RBFInterpolator(
            trnsf_input_data, output_data.T, kernel=self.kernel,
            epsilon=self.epsilon, degree=self.degree, smoothing=self.smoothing)

    def predict(self, new_input):
        if self.norm is None:
            return self.regr_model(new_input.T).T
        else:
            return self.regr_model(self.scale.transform(new_input.T)).T
