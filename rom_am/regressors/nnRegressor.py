import numpy as np
import torch
from rom_am.regressors.rom_regressor import *


class ReducRegr(torch.nn.Module):
    def __init__(self, input_dim, output_dim, layers=[20, 20, 8, 8, 4], activations=["Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid", ""]):
        super().__init__()

        assert len(layers) == len(activations)
        self.infer_lr = torch.nn.Sequential()

        for i in range(len(layers)):
            if i == 0:
                lr_in_dim = input_dim
            else:
                lr_in_dim = layers[i-1]

            lr_out_dim = layers[i]
            if i == len(layers)-1:
                lr_out_dim = output_dim

            self.infer_lr.append(torch.nn.Linear(lr_in_dim, lr_out_dim))
            if len(activations[i]) != 0:
                if activations[i] == "Sigmoid":
                    self.infer_lr.append(torch.nn.Sigmoid())
                elif activations[i] == "ReLU":
                    self.infer_lr.append(torch.nn.ReLU())

    def forward(self, x):
        inferred = self.infer_lr(x)
        return inferred


class NNRegressor(RomRegressor):

    def __init__(self, file_name, nn_layers=[40, 40, 20, 20, 4],
                 nn_activations=["Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid", ""]) -> None:
        super().__init__()
        self.nn_activations = nn_activations
        self.nn_layers = nn_layers
        self.file_name = file_name

    def predict(self, new_input):
        return self.nnModel(torch.tensor(new_input.T)).detach().numpy().T

    # For now
    def train(self, input_data, output_data, previous_input_data=None):
        super().train(input_data, output_data, previous_input_data)
        self.loadNNmodel(file_name=self.file_name,
                         layers=self.nn_layers,
                         activations=self.nn_activations)

    def loadNNmodel(self, file_name, layers, activations):
        self.nnModel = ReducRegr(
            self.input_dim, self.output_dim, layers=layers, activations=activations).double()
        self.nnModel.load_state_dict(torch.load(file_name))
        self.nnModel.eval()
