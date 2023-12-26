import numpy as np
import warnings

class RomDimensionalityReducer:
    """Baseclass for dimensionality reduction used in the latent space

    """

    def __init__(self, latent_dim) -> None:
        self.latent_dim = latent_dim
        self._reduced_data = None
        self.map_mat = None
        self.interface_dim = None
        self.tree = None
        self.hull = None
        pass

    def train(self, data, map_used=None):
        """Training the dimensionality reducer

        Parameters
        ----------
        data  : numpy.ndarray
            Snapshot matrix of data, of (N, m) size
        map_used  : numpy.ndarray or None
            Snapshot matrix of mapping indices (from interface
            nodes to all the nodes), of (N, n) size.
            If None, no mapping is used
            Default : None

        Returns
        ------

        """
        if self.latent_dim is not None:
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
        raise Exception('"encode" has to be implemented in the derived class!')

    def decode(self, new_latent_data, high_dim=False) -> np.ndarray:
        """Training the regressor

        Parameters
        ----------
        new_latent_data  : numpy.ndarray
            Snapshot matrix of data, of (r, m) size
        high_dim  : bool, optional
            Whether or not to override the mapping and returning the
            high dimensional space. Only considered when self.map_mat
            is not None
            Default : False

        Returns
        ------
        highDim_data  : numpy.ndarray
            Snapshot matrix of data,
            of (N, m) size
            or (n, m) is self.map_mat is not None

        """
        raise Exception('"decode" has to be implemented in the derived class!')

    def check_decoder_in(self, new_latent_data):
        assert (new_latent_data.shape[0] == self.latent_dim
                ), f"The dimension of the decoder input point should be  {self.latent_dim}. {new_latent_data.shape[0]} was given."

    def check_decoder_out(self, new_out_data):
        if self.map_mat is None:
            out_dim = self.high_dim
        else:
            out_dim = self.interface_dim
        assert (new_out_data.shape[0] == out_dim
                ), f"The dimension of the decoded point should be  {out_dim}. {new_out_data.shape[0]}-dimensional data was computed."

    def check_encoder_in(self, new_data):
        assert (new_data.shape[0] == self.high_dim
                ), f"The dimension of the encoder input point should be  {self.high_dim}. {new_data.shape[0]} was given."

    def check_encoder_out(self, new_encoded_data):
        assert (new_encoded_data.shape[0] == self.latent_dim
                ), f"The dimension of the encoded point should be  {self.latent_dim}. {new_encoded_data.shape[0]}-dimensional data was computed."

    def _check_encode_accuracy(self, point, encoded_):
        decode = getattr(self, "decode", None)
        if callable(decode):
            reconstructed_ = decode(encoded_, high_dim=True)
            err = np.linalg.norm(reconstructed_ - point, axis = 0)/np.linalg.norm(point, axis = 0)
            self.EncReconsErr = err
            if np.max(err) > 0.1:
                warnings.warn("The encoder isn't working well")
                return None
            else:
                return 0
        else:
            return 0

    def _check_encode_nearness(self, encoded_):
        try:
            if self.hull.find_simplex(self.minmaxScaler.transform((encoded_[:3, :].T)))<0:
                dist, _ = self.tree.query(self.minmaxScaler.transform((encoded_[:3, :].T)))
                if dist > 0.2 * self.max_dist:
                    warnings.warn("Warning, the new point is too far")
        except AttributeError:
            pass


    @property
    def reduced_data(self):
        """Representations of the training data in the latent space

        Parameters
        ----------

        Returns
        ------
        reduced_data  : numpy.ndarray
            Matrix of data, of (r, m) size

        """
        if self._reduced_data is None:
            raise Exception(
                '"reduced_data" has to be implemented in the derived class!')
        else:
            return self._reduced_data
