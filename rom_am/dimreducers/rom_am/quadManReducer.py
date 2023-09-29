import numpy as np
from rom_am import QUAD_MAN
from rom_am.dimreducers.rom_am.podReducer import PodReducer


class QuadManReducer(PodReducer):

    def train(self, data, normalize=True, center=True, map_used=None):
        super().train(data, normalize, center, map_used)
        if map_used is not None:
            self.inverse_project_Vbar = self.map_mat @ self.rom.denormalize(
                self.pod.Vbar)

    def _call_POD_core(self):
        return QUAD_MAN()

    def _mapped_decode(self, new_data):
        return super()._mapped_decode(new_data) + self.inverse_project_Vbar @ self.pod._kron_x_sq(new_data)
