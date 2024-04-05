from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from rom_am.regressors.rom_regressor import *
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.subsample import Subsample
from copy import deepcopy


class MapieSolidRegressor(RomRegressor):

    def __init__(self, regul_alpha, poly_degree, mapie_alpha=0.1) -> None:
        super().__init__()
        self.regul_alpha = regul_alpha
        self.poly_degree = poly_degree
        self.mapie_alpha = mapie_alpha

    def train(self, input_data, output_data):
        super().train(input_data, output_data)

        regr_model = make_pipeline(
            PolynomialFeatures(self.poly_degree), Ridge(alpha=self.regul_alpha))
        self.regr_model = []
        for i in range(self.output_dim):
            mapie = MapieRegressor(regr_model,
                                   method="base",
                                   cv=None)
            mapie.fit(input_data.T, output_data[i, :])
            self.regr_model.append(deepcopy(mapie))
            del mapie

    def predict(self, new_input, alpha=None):
        res = np.empty((self.output_dim, new_input.shape[1]))
        std = np.empty((self.output_dim, 2, new_input.shape[1]))

        if alpha is None:
            alpha = self.mapie_alpha
        for i in range(self.output_dim):
            res_, std_ = self.regr_model[i].predict(new_input.T, alpha = alpha)
            res[i, :] = res_.copy()
            std[i, 0, :] = std_[:, 0, 0].copy()
            std[i, 1, :] = std_[:, 1, 0].copy()

        self.std = std
        return res
