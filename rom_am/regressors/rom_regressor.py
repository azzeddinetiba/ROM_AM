import numpy as np


class RomRegressor:
    """Baseclass for regression used in the latent space

    """

    def __init__(self) -> None:
        pass

    def train(self, input_data, output_data):
        """Training the regressor

        Parameters
        ----------
        input_data  : numpy.ndarray
            Snapshot matrix of input data, of (Nin, m) size
        output_data : numpy.ndarray
            Second Snapshot matrix data, of (Nout, m) size
            output data

        Returns
        ------

        """
        assert (input_data.shape[1] == output_data.shape[1]
                ), "Training data has to have the same number of samples in input and output."
        self.input_dim = input_data.shape[0]
        self.output_dim = output_data.shape[0]
        pass

    def predict(self, new_input,) -> np.ndarray:
        """Regressor prediction

        Parameters
        ----------
        new_input  : numpy.ndarray
            New inputs matrix, of (Nin, m) size

        Returns
        ------
        output_result : numpy.ndarray
            Solution matrix data, of (Nout, m) size

        """
        raise Exception(
            '"Predict" has to be implemented in the derived class!')

    def check_predict_in(self, new_input):
        assert (new_input.shape[0] == self.input_dim
                ), f"The dimension of the regression input points should be {self.input_dim}. {new_input.shape[0]} was given."

    def check_predict_out(self, output):
        assert (output.shape[0] == self.output_dim
                ), f"The regression output should be of dimension {self.output_dim}. {output.shape[0]}-dimensional data was computed."
