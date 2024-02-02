import numpy as np
from rom_am.quad_man import QUAD_MAN
from rom_am.dimreducers.rom_am.podReducer import PodReducer


class QuadManReducer(PodReducer):

    def train(self, data, map_used=None, normalize=True, center=True, alg="svd", to_copy=True, to_copy_order='F'):
        super().train(data, map_used, normalize, center, alg=alg,
                      to_copy=to_copy, to_copy_order=to_copy_order)

    def _call_POD_core(self):
        return QUAD_MAN()

    def _mapped_decode(self, new_data):
        return super()._mapped_decode(new_data) + self.pod.Vbar[self.map_mat, :] @ self.pod._kron_x_sq(new_data)

    def encode(self, new_data):

        interm = self.rom.normalize(self.rom.center(new_data))
        encoded_ = self.pod.modes.T @ interm
        # self._check_encode_nearness(encoded_)
        # accurate_ = self._check_encode_accuracy(new_data, encoded_)
        # if accurate_ is None:
        if False:
            return None
        else:
            return encoded_
