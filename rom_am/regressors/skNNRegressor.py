from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from rom_am.regressors.rom_regressor import *


class SKNNRegressor(RomRegressor):

    def __init__(self, hidden_layers=np.array([
            40, 40, 40]), norm=None) -> None:
        super().__init__()
        self.hidden_layers = hidden_layers
        self.norm = norm

    def train(self, input_data, output_data, weights):
        super().train(input_data, output_data)
        if self.norm is None:
            scale = None
        elif self.norm == "max":
            scale = MaxAbsScaler()
        elif self.norm == "std":
            scale = StandardScaler()

        regr_model_ = MLPRegressor(
            random_state=1, hidden_layer_sizes=self.hidden_layers,
            solver ='adam', activation='relu', warm_start=True,
            max_iter=2500, tol=0.00001)
        self.regr_model = make_pipeline(scale, regr_model_)
        self.regr_model.fit(input_data.T, output_data.T)

    def predict(self, new_input):
        return self.regr_model.predict(new_input.T).T

    def update(self, input_data, output_data, weights):
        self.train(input_data.T, output_data.T)