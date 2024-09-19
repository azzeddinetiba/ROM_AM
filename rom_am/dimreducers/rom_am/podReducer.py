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

    def train(self, data, map_used=None, normalize=True, center=True, alg="svd", to_copy=True, to_copy_order='F'):

        super().train(data, map_used)

        pod = self._call_POD_core()
        rom = ROM(pod)

        rom.decompose(X=data, normalize=normalize, center=center,
                      rank=self.latent_dim, alg=alg, to_copy=to_copy, to_copy_order=to_copy_order)

        self.latent_dim = pod.kept_rank
        self.normalize = normalize
        self.center = center
        self.rom = rom
        self.pod = pod

        if map_used is not None:
            if map_used.dtype == int:
                self.interface_dim = map_used.shape[0]
            else:
                self.interface_dim = len(np.argwhere(map_used))
            self.map_mat = map_used
            self.mapped_modes = self.pod.modes[self.map_mat, :]

        self.minmaxScaler = MinMaxScaler()
        self.minmaxScaler.fit(self.pod.pod_coeff[:3, :].T)
        self.tree = KDTree(self.minmaxScaler.transform(
            self.pod.pod_coeff[:3, :].T))
        self.hull = Delaunay(self.minmaxScaler.transform(
            self.pod.pod_coeff[:3, :].T))
        dists, _ = self.tree.query(self.minmaxScaler.transform(
            self.pod.pod_coeff[:3, :].T), k=self.pod.pod_coeff.shape[1])
        self.max_dist = dists.max()

    def encode(self, new_data, high_dim=True):

        if high_dim or self.map_mat is None:
            interm = self.rom.normalize(self.rom.center(new_data))
            encoded_ = self.pod.project(interm)
        else:
            interm = self.rom.normalize(self.rom.center(
                new_data, self.map_mat), self.map_mat)
            encoded_ = self.mapped_modes.T @ interm

        # Future devs
        # self._check_encode_nearness(encoded_)
        # accurate_ = self._check_encode_accuracy(new_data, encoded_)
        # if accurate_ is None:
        #     return None
        # else:
        #     return encoded_
        return encoded_

    def decode(self, new_data, high_dim=False):

        if self.map_mat is not None and not high_dim:
            interm = self._mapped_decode(new_data)
            return self.rom.decenter(self.rom.denormalize(interm, self.map_mat), self.map_mat)
        else:
            interm = self.pod.inverse_project(new_data)
            return self.rom.decenter(self.rom.denormalize(interm))

    def _call_POD_core(self, ):
        return POD()

    def _mapped_decode(self, new_data):
        return self.mapped_modes @ new_data

    @property
    def reduced_data(self):
        return self.pod.pod_coeff

    def truncate(self, new_dim):
        self.pod._truncate(new_dim)
        self.latent_dim = new_dim
