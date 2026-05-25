import os
import numpy as np
from rom_am.hodmd import HODMD


"""
Tests on High Order DMD based on the nonlinear spring example (with the velocity component,
constant pressure load) studied in the eDMD_Spring example
"""

X = np.load("./tests/DATA/hodmd_train_X.npy")
Y = np.load("./tests/DATA/hodmd_train_Y.npy")

dt = 6e-4
hodmd = HODMD()
hodmd.decompose(X, Y=Y, dt=dt, hod=100,)


def test_hodmd_decomposition():
    assert hodmd.modes is not None


pred_t = np.load("./tests/DATA/hodmd_pred_t.npy")
pred_ref = np.load("./tests/DATA/hodmd_pred_res.npy")

pred = np.real(hodmd.predict(pred_t,))


def test_hodmd_prediction():
    assert pred.shape == pred_ref.shape

    n_early = min(100, pred.shape[1])
    assert np.allclose(pred[:, :n_early], pred_ref[:, :n_early], rtol=0.1, atol=8e-2)

    rel_l2 = np.linalg.norm(pred - pred_ref) / max(np.linalg.norm(pred_ref), np.finfo(float).eps)
    assert rel_l2 < 0.5
