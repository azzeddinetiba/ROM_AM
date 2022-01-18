import numpy as np
import scipy.linalg as sp


class ROM:
    def __init__(self, rom_object):

        self.model = rom_object
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
            opt_trunc=False,
            tikhonov=0,
            *args,
            **kwargs,):

        self.snapshots = X.copy()

        u, s, vh = self.model.decompose(X=self.snapshots,
                                        center=center,
                                        alg=alg,
                                        rank=rank,
                                        opt_trunc=opt_trunc,
                                        tikhonov=tikhonov,
                                        *args,
                                        **kwargs,)

        self.singvals = s
        self.modes = u
        self.time = vh

    def predict(self, t, init=0, t1=0, *args, **kwargs):
        return self.model.predict(t=t, init=init, t1=t1, *args, **kwargs)

    def reconstruct(self, rank=None):
        return self.model.reconstruct(rank=rank)
