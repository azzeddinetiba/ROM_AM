import pytest
import numpy as np
from rom_am import DMD


def func(nx, nt, t_stop=2):
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 2, ny)
    tt, xx = np.meshgrid(t, x, )
    z = np.exp(-np.abs((xx-.5)*(tt-1))) + np.sin(xx**tt)

    return z


nx = 2000
ny = 100
x = np.linspace(0, 1, nx)
t = np.linspace(0, 2, ny)
tt, xx = np.meshgrid(t, x, )
z = func(nx, ny)

dt = t[-1]-t[-2]
dmd = DMD()
dmd.decompose(z[:, :-1], Y=z[:, 1::], dt=dt, rank=0, sorting="real")


correct_lambd = np.array([1.00819927e+00-0.01295502j,  1.00819927e+00+0.01295502j, 1.00274625e+00-0.02680876j,
                         1.00274625e+00+0.02680876j, 9.98114433e-01+0.j,  9.79691360e-01-0.05881754j,
                         9.79691360e-01+0.05881754j,  9.47531458e-01-0.09433631j, 9.47531458e-01+0.09433631j,  9.00659235e-01 +
                         0.12695041j, 9.00659235e-01-0.12695041j,  8.31605128e-01 -
                          0.16034071j, 8.31605128e-01+0.16034071j,
                         7.68087673e-01+0.j, 7.20461365e-01-0.18799393j,  7.20461365e-01 +
                          0.18799393j, 5.43869552e-01-0.20607373j,
                         5.43869552e-01+0.20607373j, 4.98983607e-01+0.j, -7.98857162e-08+0.j])

correct_eigval = np.array([4.08295412e-01 - 0.6360235j,  4.08295412e-01 + 0.6360235j,
                           1.53437632e-01 - 1.32308414j,  1.53437632e-01 + 1.32308414j,
                           -9.34236486e-02 + 0.j, -9.26576998e-01 - 2.96825908j,
                           -9.26576998e-01 + 2.96825908j, -2.42369021e+00 - 4.91203684j,
                           -2.42369021e+00 + 4.91203684j, -4.69219670e+00 + 6.93149859j,
                           -4.69219670e+00 - 6.93149859j, -8.22428272e+00 - 9.42833391j,
                           -8.22428272e+00 + 9.42833391j, -1.30606440e+01 + 0.j,
                           -1.45989725e+01 - 12.63458314j, -1.45989725e+01 + 12.63458314j,
                           -2.68274916e+01 - 17.92830409j, -2.68274916e+01 + 17.92830409j,
                           -3.44115107e+01 + 0.j, -8.08962104e+02+155.50883635j])


def test_dmd_decomposition():
    dmd_ = DMD()
    dmd_.decompose(z[:, :-1], Y=z[:, 1::], dt=dt, rank=0, sorting="real")
    assert dmd_.dmd_modes is not None and dmd_.A_tilde is not None


@pytest.mark.parametrize('rank', np.linspace(1, 20, 5, dtype=int))
def test_trunc(rank):
    dmd_ = DMD()
    dmd_.decompose(z[:, :-1], Y=z[:, 1::], dt=dt, rank=rank, sorting="real")
    assert dmd_.dmd_modes.shape == (nx, rank) and dmd_.lambd.shape == (
        rank, ) and dmd_.A_tilde.shape == (rank, rank)


def test_trunc_2():
    dmd_ = DMD()
    dmd_.decompose(z[:, :-1], Y=z[:, 1::], dt=dt, rank=0, sorting="real")
    assert dmd_.dmd_modes.shape == (nx, 20)


def test_eig_form_dmd():
    assert dmd.lambd.shape == (20, ) and dmd.eigenvalues.shape == (20, )


def test_eig_dmd():
    assert np.allclose(dmd.lambd, correct_lambd) and np.allclose(
        dmd.eigenvalues, correct_eigval)
