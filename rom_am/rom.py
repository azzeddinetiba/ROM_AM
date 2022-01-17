import numpy as np
import scipy.linalg as sp


class ROM:
    def __init__(self, rom):

        self.model = rom
        self.snapshots = None

        self.singvals = None
        self.modes = None
        self.time = None

    def decompose(
        self,
        X,
        center=False,
        alg="svd",
        rank=0,
        sorting="abs",
        opt_trunc=False,
        tikhonov=0,
        *args,
        **kwargs,
    ):

        self.snapshots = X.copy()

        u, s, vh = self.model.decompose(self,
                                        X=self.snapshots,
                                        center=center,
                                        alg=alg,
                                        rank=rank,
                                        sorting=sorting,
                                        opt_trunc=opt_trunc,
                                        tikhonov=tikhonov,
                                        *args,
                                        **kwargs,)

        self.singvals = s
        self.modes = u
        self.time = vh

        # if self.rom == "dmdc":
        #     u_til_1, u_til_2, s_til, vh_til, lambd, phi = self._dmdc_decompose(
        #         X, Y, Y_input, rank
        #     )
        #     u = u_til_1
        #     vh = vh_til
        #     s = s_til

    def predict(self, t, init=0, t1=0, *args, **kwargs):
        return self.model.predict(t=t, init=init, t1=t1, *args, **kwargs)

    def reconstruct(self, rank):
        return self.model.reconstruct(rank=rank)
