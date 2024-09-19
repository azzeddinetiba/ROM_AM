import numpy as np
from rom_am.quad_man import QUAD_MAN
from rom_am.dimreducers.rom_am.podReducer import PodReducer


class QuadManReducer(PodReducer):

    def train(self, data, map_used=None, normalize=True, center=True, alg="svd", to_copy=True, to_copy_order='F'):
        super().train(data, map_used, normalize, center, alg=alg,
                      to_copy=to_copy, to_copy_order=to_copy_order)
        if map_used is not None:
            self.mapped_Vbar = self.pod.Vbar[self.map_mat, :]

    def _call_POD_core(self):
        return QUAD_MAN()

    def _mapped_decode(self, new_data):
        return super()._mapped_decode(new_data) + self.mapped_Vbar @ self.pod._kron_x_sq(new_data)

    def encode(self, new_data, high_dim=True):

        if high_dim or self.map_mat is None:
            interm = self.rom.normalize(self.rom.center(new_data))
            encoded_ = self.pod.modes.T @ interm
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
