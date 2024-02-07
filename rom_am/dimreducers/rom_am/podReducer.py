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
        self.space_computed = False
        self.ids_to_correct = None
        self.corrected_points = []
        self.far_points = []

    def train(self, data, map_used=None, normalize=True, center=True, alg="svd", to_copy=True,
              to_copy_order='F', compute_space=False):

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

        if compute_space:
            self.minmaxScaler = MinMaxScaler()
            self.minmaxScaler.fit(self.pod.pod_coeff.T)
            self.tree = KDTree(self.minmaxScaler.transform(
                self.pod.pod_coeff.T))
            # self.hull = Delaunay(self.minmaxScaler.transform(
            #    self.pod.pod_coeff[:3, :].T))
            self.meanScaledPoint = self.minmaxScaler.transform(
                self.pod.pod_coeff.mean(axis=1).reshape((1, -1)))
            dists, _ = self.tree.query(self.meanScaledPoint,
                                       k=self.pod.pod_coeff.shape[1])
            self.max_dist = dists.max()
            self.space_computed = True

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

    def _correct_point(self, new_points, correc_coeff=0.2, cutoff=0.2):
        dists_to_mean = np.linalg.norm(self.minmaxScaler.transform(
            new_points.T) - self.meanScaledPoint, axis=1)
        ids_to_correct = np.argwhere(
            dists_to_mean > (1+cutoff) * self.max_dist).ravel()
        self.ids_to_correct = ids_to_correct.copy()
        if len(ids_to_correct) > 0:
            warnings.warn("Warning, the new point is too far")
            self.far_points.append(new_points[:, ids_to_correct].copy())
            newScaledPoints = self.minmaxScaler.transform(new_points[:, ids_to_correct].T)
            newScaledPoints += correc_coeff * (self.meanScaledPoint - newScaledPoints)
            new_points[:, ids_to_correct] = self.minmaxScaler.inverse_transform(newScaledPoints).T
            self.corrected_points.append(new_points[:, ids_to_correct].copy())

    def _decode(self, new_data, high_dim=False):
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
