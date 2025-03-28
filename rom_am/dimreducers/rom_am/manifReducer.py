import numpy as np
from rom_am.pod import POD
from rom_am.rom import ROM
from rom_am.dimreducers.rom_DimensionalityReducer import *
from rom_am.dimreducers.rom_am.podReducer import PodReducer
from rom_am import utils
from scipy.interpolate import RBFInterpolator


class ManifInterpReducer(RomDimensionalityReducer):

    def __init__(self, latent_dim, ) -> None:
        super().__init__(latent_dim)
        self.parametric = True
        self.weights = None
        self.calibrationQ = None
        self.pod = None
        self._m = 2
        self.nx = None

    def train(self, bases_list: list[PodReducer], params, map_used=None, normalize=True, center=True, alg="svd", to_copy=True, to_copy_order='F', refP=0, kernel='thin_plate_spline', epsilon=None,
              precomp_mean=None,
              precomp_std=None):

        ## Checks -----------------------------------------------------------------------------------------------------------------------
        self._p = len(bases_list)
        assert (self._p != 0), "The list of bases is empty. "
        assert (
            self._p == params.shape[1]), "The number of parameter points is different than the fed bases. "
        self.high_dim = bases_list[0].high_dim
        for i in range(self._p):
            assert (self.latent_dim ==
                    bases_list[i].latent_dim), "The latent dimension of the input base is different than the target latent dimension. "
            assert (
                self.high_dim == bases_list[i].high_dim), "All the input bases should belong to the same high dimensional space. "
        ##  -----------------------------------------------------------------------------------------------------------------------

        self.pod = POD()
        self.rom = ROM(self.pod)
        self.rom.nx = bases_list[0].rom.nx
        iref = 0  # For now
        self.reducRef = bases_list[iref]
        stacked_U_log = np.empty(
            (self._p, self.high_dim, self.latent_dim))
        stacked_romNorms = np.empty(
            (self.high_dim, self._p))
        stacked_romMeans = np.empty(
            (self.high_dim, self._p))

        stacked_U_log[iref, :, :] = np.zeros((self.high_dim, self.latent_dim))
        # stacked_romNorms[:, iref] = bases_list[iref].rom.snap_norms
        # stacked_romMeans[:, iref] = bases_list[iref].rom.mean_flow
        for i in np.delete(np.arange(0, self._p), iref):
            stacked_U_log[i, :, :] = utils.log_U(
                bases_list[iref].pod.modes, bases_list[i].pod.modes)
            # stacked_romNorms[:, i] = bases_list[i].rom.snap_norms
            # stacked_romMeans[:, i] = bases_list[i].rom.mean_flow

        if precomp_std is not None:
            self.rom.normalize_ = True
            self.rom.normalization = "norm"
            # self.rom.snap_norms = stacked_romNorms.mean(axis=1)
            self.rom.snap_norms = precomp_std.copy()
            self.rom.zeroIds = np.argwhere(np.isclose(
                self.rom.snap_norms, 0, atol=1e-12)).ravel()
            self.rom.snap_norms = np.where(np.isclose(
                self.rom.snap_norms, 0, atol=1e-12), 1, self.rom.snap_norms)

        if precomp_mean is not None:
            self.rom.center_ = True
            # self.rom.mean_flow = stacked_romMeans.mean(axis=1)
            self.rom.mean_flow = precomp_mean.copy()

        # TODO normalize params ? same normalization as in regressors ?
        self.f_U = RBFInterpolator(
            params.T, stacked_U_log, kernel=kernel, epsilon=epsilon)

    def predictNewModes(self, new_mu, bases_list: list[PodReducer]):
        U_pred = utils.exp_U(self.reducRef.pod.modes,
                             self.f_U(new_mu.T)[0, :, :])
        self.pod.modes = U_pred

        self._orientationsAdjustment(bases_list, U_pred)
        # self._findCalibrationMatrix(bases_list, U_pred)

    # def _findCalibrationMatrix(self, bases_list: list[PodReducer], measureBase):

    #     weightedSum = (np.array(
    #         [a.pod.modes for a in bases_list]).T.dot(self.weights)).T
    #     minimizationMatrix = measureBase.T @ weightedSum
    #     prod = POD()
    #     u, _, vh = prod.decompose(minimizationMatrix, thin=False)
    #     self.calibrationQ = u @ vh

    def _orientationsAdjustment(self, bases_list: list[PodReducer], measureBase):

        idClosestBase, _, distsToPredictedBase = utils.minDistBase(
            [a.pod.modes for a in bases_list], measureBase)
        self.weights = np.array(distsToPredictedBase)**(-self._m)
        self.weights /= np.sum(self.weights)

        for i in range(self._p):
            if i == idClosestBase:
                continue
            for j in range(self.latent_dim):
                condition = np.linalg.norm(bases_list[i].pod.modes[:, idClosestBase] - bases_list[i].pod.modes[:, j]) > np.linalg.norm(
                    bases_list[i].pod.modes[:, idClosestBase] + bases_list[i].pod.modes[:, j])
                if condition:
                    bases_list[i].pod.invertOrientation(j)


    def encode(self, new_data):

        # if high_dim or self.map_mat is None:
        interm = self.rom.normalize(self.rom.center(new_data))
        encoded_ = self.pod.project(interm)
        # else:
        #     interm = self.rom.normalize(self.rom.center(
        #         new_data, self.map_mat), self.map_mat)
        #     encoded_ = self.mapped_modes.T @ interm

        return encoded_

    def decode(self, new_data, high_dim=False):
        try:
            interm = self.pod.inverse_project(new_data)
            return self.rom.decenter(self.rom.denormalize(interm))
        except:
            raise Exception(
                "The basis for the unseen parameter is still not computed, ’predictNewModes’ should be called.")

    def _call_POD_core(self, ):
        return POD()
