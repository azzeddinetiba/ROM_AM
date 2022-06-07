import os
import numpy as np
from rom_am import KERDMD


"""
Tests on Kernel DMD based on a nonlinear dynamic system 
(See The 'Koopman.ipynb', example)
"""

X = np.load("./tests/DATA/kerdmd_train_X.npy")
Y = np.load("./tests/DATA/kerdmd_train_Y.npy")

dt = 0.06
kdmd = KERDMD()
kdmd.decompose(X, Y=Y, dt=dt, kernel = "poly", p=2)


def test_kerdmd_decomposition():
    assert kdmd.dmd_modes is not None


pred_t = np.load("./tests/DATA/kerdmd_test_t.npy")
pred_ref = X.copy()

pred = np.real(kdmd.predict(pred_t, method=3))


def test_kerdmd_prediction():
    assert np.allclose(pred, pred_ref, rtol=0.01, atol=1e-3)
