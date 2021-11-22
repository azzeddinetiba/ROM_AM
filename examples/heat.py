# -*- coding: utf-8 -*-
"""
%
%          TIBA Azzeddine
"""

import numpy as np
import scipy.sparse.linalg
import scipy.sparse as sp

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import font_manager as fm, rcParams


class FEM:

    def __init__(self, N, Nt, length, T, kappa = 1):

        """

        :param N: int
            number of elements (spatial)
        :param Nt: int
            number of time steps
        :param length: float
            length of the spatial domain
        :param T: float
            total time period
        """

        self.N = N
        self.Nt = Nt
        self.h = length / self.N
        self.dt = T / (self.Nt - 1)
        self.U = np.zeros((N+1, Nt))
        self.kappa = kappa

    def K(self, ):
        v = np.ones((self.N))
        K1 = sp.diags(-v, 1)
        K_1 = sp.diags(-v, -1)
        KD = 2 * sp.eye(self.N + 1, self.N + 1).tocsc()
        KD[0, 0] = 1
        KD[self.N, self.N] = 1
        return ((K1 + K_1).tocsc() + KD) * (1 / self.h)

    def M(self, ):
        v = np.ones((self.N))
        M1 = sp.diags(v, 1)
        M_1 = sp.diags(v, -1)
        MD = 4 * sp.eye(self.N + 1, self.N + 1).tocsc()
        MD[0, 0] = 2
        MD[self.N, self.N] = 2
        return ((M1 + M_1).tocsc() + MD) * self.h / 6

    def solve(self, t_ini, neumann):
        """

        :param t_ini:
        :param neumann:
        :return:
        """

        self.U[:, 0] = t_ini
        self.M_mat = self.M()
        self.K_mat = self.K()

        self.A = self.M_mat + self.dt * self.kappa * self.K_mat
        self.A_lu = sp.linalg.splu(self.A)
        neu = np.zeros(self.N+1)

        for i in range(1, self.Nt):

            neu[0] = neumann(i * self.dt)
            rhs = self.M_mat @ self.U[:, i-1] + self.dt * neu
            self.U[:, i] = self.A_lu.solve(rhs)

        return self.U
