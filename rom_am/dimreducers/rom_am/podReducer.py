import numpy as np
from rom_am.pod import POD
from rom_am.rom import ROM
from rom_am.dimreducers.rom_DimensionalityReducer import *


class PodReducer(RomDimensionalityReducer):

    def __init__(self, latent_dim, ) -> None:
        super().__init__(latent_dim)

    def train(self, data, map_used=None, normalize=True, center=True):

        super().train(data, map_used)

        pod = self._call_POD_core()
        rom = ROM(pod)
        rom.decompose(X=data, normalize=normalize, center=center,
                      rank=self.latent_dim)

        self.latent_dim = pod.kept_rank
        self.normalize = normalize
        self.center = center
        self.rom = rom
        self.pod = pod

        if map_used is not None:
            self.interface_dim = map_used.shape[0]
            self.map_mat = map_used
            self.inverse_project_mat = self.map_mat @ self.rom.denormalize(
                self.pod.modes)

            if center:
                self.mapped_mean_flow = self.map_mat @ self.rom.mean_flow.reshape(
                    (-1, 1))

    def encode(self, new_data):

        interm = self.rom.normalize(self.rom.center(new_data))
        return self.pod.project(interm)

    def decode(self, new_data, high_dim=False):

        if self.map_mat is not None and not high_dim:
            interm = self._mapped_decode(new_data)
            if self.center:
                interm = (
                    interm + self.mapped_mean_flow)
            return interm

        else:

            interm = self.pod.inverse_project(new_data)
            return self.rom.decenter(self.rom.denormalize(interm))

    def _call_POD_core(self, ):
        return POD()

    def _mapped_decode(self, new_data):
        return self.inverse_project_mat @ new_data

    @property
    def reduced_data(self):
        return self.pod.pod_coeff
