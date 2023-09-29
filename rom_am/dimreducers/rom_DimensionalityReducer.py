import numpy as np


class RomDimensionalityReducer:
    """Baseclass for dimensionality reduction used in the latent space

    """

    def __init__(self, latent_dim) -> None:
        self.latent_dim = latent_dim
        self._reduced_data = None
        pass

    def train(self, data):
        """Training the dimensionality reducer

        Parameters
        ----------
        data  : numpy.ndarray
            Snapshot matrix of data, of (N, m) size

        Returns
        ------

        """
        assert (data.shape[0] >= self.latent_dim
                ), "Training data has to have a bigger dimension than the DimReducer latent dimension."
        self.high_dim = data.shape[0]
        pass

    def encode(self, new_data) -> np.ndarray:
        """Project the data instances to the latent space

        Parameters
        ----------
        new_data  : numpy.ndarray
            Snapshot matrix of data, of (N, m) size

        Returns
        ------
        latent_data  : numpy.ndarray
            Snapshot matrix of data, of (r, m) size

        """
        self._check_encoder(new_data)
        raise Exception('"encode" has to be implemented in the derived class!')

    def _check_encoder(self, new_data):
        assert (new_data.shape[0] == self.high_dim
                ), f"The dimension of the encoder input point should be  {self.high_dim}. {new_data.shape[0]} was given."

    def decode(self, new_latent_data) -> np.ndarray:
        """Training the regressor

        Parameters
        ----------
        new_latent_data  : numpy.ndarray
            Snapshot matrix of data, of (r, m) size

        Returns
        ------
        highDim_data  : numpy.ndarray
            Snapshot matrix of data, of (N, m) size

        """
        self._check_decoder(new_latent_data)
        raise Exception('"decode" has to be implemented in the derived class!')

    def _check_decoder(self, new_latent_data):
        assert (new_latent_data.shape[0] == self.latent_dim
                ), f"The dimension of the decoder input point should be  {self.latent_dim}. {new_latent_data.shape[0]} was given."

    @property
    def reduced_data(self):
        if self._reduced_data is None:
            raise Exception('"reduced_data" has to be implemented in the derived class!')
        else:
            return self._reduced_data
