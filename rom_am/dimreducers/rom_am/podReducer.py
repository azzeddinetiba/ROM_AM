import numpy as np
from rom_am.pod import POD
from rom_am.rom import ROM
from rom_am.dimreducers.rom_DimensionalityReducer import *
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
from sklearn.preprocessing import MinMaxScaler


class PodReducer(RomDimensionalityReducer):

    def __init__(self, latent_dim, ) -> None:
        super().__init__(latent_dim)

    def train(self, data, map_used=None, normalize=True, center=True, alg="svd", to_copy=True):

        super().train(data, map_used)

        pod = self._call_POD_core()
        rom = ROM(pod)

        rom.decompose(X=data, normalize=normalize, center=center,
                      rank=self.latent_dim, alg=alg, to_copy=to_copy)

        self.latent_dim = pod.kept_rank
        self.normalize = normalize
        self.center = center
        self.rom = rom
        self.pod = pod

        if map_used is not None:
            self.interface_dim = map_used.shape[0]
            self.map_mat = map_used

        """
        self.minmaxScaler = MinMaxScaler()
        self.minmaxScaler.fit(self.pod.pod_coeff[:3, :].T)
        self.tree = KDTree(self.minmaxScaler.transform(
            self.pod.pod_coeff[:3, :].T))
        self.hull = Delaunay(self.minmaxScaler.transform(
            self.pod.pod_coeff[:3, :].T))
        dists, _ = self.tree.query(self.minmaxScaler.transform(
            self.pod.pod_coeff[:3, :].T), k=self.pod.pod_coeff.shape[1])
        self.max_dist = dists.max()
        """

    def encode(self, new_data):

        interm = self.rom.normalize(self.rom.center(new_data))
        encoded_ = self.pod.project(interm)
        # self._check_encode_nearness(encoded_)
        # accurate_ = self._check_encode_accuracy(new_data, encoded_)
        # if accurate_ is None:
        if False:
            return None
        else:
            return encoded_

    def decode(self, new_data, high_dim=False):

        if self.map_mat is not None and not high_dim:
            interm = self._mapped_decode(new_data)
            if self.center:
                interm = (
                    interm + self.rom.mean_flow[self.map_mat].reshape(
                        (-1, 1)))
            return interm

        else:

            interm = self.pod.inverse_project(new_data)
            return self.rom.decenter(self.rom.denormalize(interm))

    def _call_POD_core(self, ):
        return POD()

    def _mapped_decode(self, new_data):
        return self.rom.denormalize(
            self.pod.modes)[self.map_mat, :] @ new_data

    @property
    def reduced_data(self):
        return self.pod.pod_coeff
