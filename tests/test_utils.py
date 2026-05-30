"""
Tests for rom_am/utils.py 
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from helpers import _random_orthonormal

from rom_am.utils import (
    log_A,
    exp_A,
    log_U,
    exp_U,
    angles,
    dist,
    minDistBase,
    rank1_update,
    cnvx_nnls,
    _determine_pod_alg_square_matrices,
)


class TestLogExpA:

    def test_log_A_linear(self):
        """Without exp flag: log_A returns the difference."""
        rng = np.random.default_rng(1)
        A  = rng.standard_normal((5, 5))
        At = rng.standard_normal((5, 5))
        result = log_A(A, At, exp=False)
        assert_allclose(result, At - A)

    def test_log_A_matrix_exp(self):
        """With exp flag: result is matrix logarithm-based."""
        rng = np.random.default_rng(2)
        B = rng.standard_normal((4, 4))
        A  = B @ B.T + 4 * np.eye(4)
        At = A + 0.01 * np.eye(4)
        result = log_A(A, At, exp=True)
        assert result.shape == (4, 4)
        assert np.isfinite(result).all()

    def test_exp_A_linear(self):
        """Without exp flag: exp_A returns the sum."""
        rng = np.random.default_rng(3)
        A  = rng.standard_normal((5, 5))
        At = rng.standard_normal((5, 5))
        result = exp_A(A, At, exp=False)
        assert_allclose(result, A + At)

    def test_exp_A_matrix_exp(self):
        """With exp flag: result is expm-based."""
        rng = np.random.default_rng(4)
        B  = rng.standard_normal((4, 4))
        A  = B @ B.T + 4 * np.eye(4)
        At = 0.01 * np.eye(4)
        result = exp_A(A, At, exp=True)
        assert result.shape == (4, 4)
        assert np.isfinite(result).all()

    def test_log_exp_A_roundtrip(self):
        """exp_A(A, log_A(A, At, exp=True), exp=True) ≈ At."""
        rng = np.random.default_rng(5)
        B  = rng.standard_normal((4, 4))
        A  = B @ B.T + 4 * np.eye(4)
        At = A + 0.02 * np.eye(4)
        delta = log_A(A, At, exp=True)
        reconstructed = exp_A(A, delta, exp=True)
        assert_allclose(reconstructed, At, atol=1e-10)


class TestLogExpU:

    def test_log_U_shape(self):
        """log_U returns an (N, k) matrix."""
        N, k = 20, 3
        U  = _random_orthonormal(N, k, seed=10)
        Ut = _random_orthonormal(N, k, seed=11)
        delta = log_U(U, Ut)
        assert delta.shape == (N, k)

    def test_exp_U_orthonormality(self):
        """exp_U produces an approximately orthonormal result."""
        N, k = 20, 3
        U  = _random_orthonormal(N, k, seed=12)
        Ut = _random_orthonormal(N, k, seed=13)
        delt = log_U(U, Ut)
        V = exp_U(U, delt)
        assert V.shape == (N, k)
        gram = V.T @ V
        assert_allclose(gram, np.eye(k), atol=1e-10)

    def test_log_exp_U_roundtrip(self):
        """exp_U(U, log_U(U, Ut)) ≈ Ut (up to sign convention)."""
        N, k = 20, 3
        U  = _random_orthonormal(N, k, seed=14)
        Ut = _random_orthonormal(N, k, seed=15)
        delt = log_U(U, Ut)
        V = exp_U(U, delt)
        d, _ = dist(V, Ut)
        assert d < 1e-6


class TestAnglesAndDist:

    def test_angles_identical_bases(self):
        """Identical bases have near-zero principal angles."""
        U = _random_orthonormal(30, 4, seed=20)
        ang, _ = angles(U, U)
        assert np.all(ang < 1e-6)

    def test_angles_orthogonal_bases(self):
        """Fully orthogonal bases have all principal angles = pi/2."""
        n, k = 10, 3
        Q, _ = np.linalg.qr(np.random.default_rng(21).standard_normal((n, n)))
        U = Q[:, :k]
        V = Q[:, k:2*k]
        ang, _ = angles(U, V)
        assert_allclose(ang, np.full(k, np.pi / 2), atol=1e-10)

    def test_angles_shape(self):
        U = _random_orthonormal(30, 4, seed=22)
        V = _random_orthonormal(30, 4, seed=23)
        ang, _ = angles(U, V)
        assert ang.ndim == 1

    def test_dist_identical_bases(self):
        """Distance between identical bases is near zero."""
        U = _random_orthonormal(30, 4, seed=24)
        d, _ = dist(U, U)
        assert d < 1e-6

    def test_dist_positive(self):
        """Distance between different bases is positive."""
        U = _random_orthonormal(30, 4, seed=25)
        V = _random_orthonormal(30, 4, seed=26)
        d, _ = dist(U, V)
        assert d > 0.0

    def test_dist_symmetry(self):
        """dist(U, V) == dist(V, U)."""
        U = _random_orthonormal(30, 4, seed=27)
        V = _random_orthonormal(30, 4, seed=28)
        d1, _ = dist(U, V)
        d2, _ = dist(V, U)
        assert_allclose(d1, d2, atol=1e-12)

    def test_angles_with_alg_options(self):
        U = _random_orthonormal(30, 4, seed=29)
        V = _random_orthonormal(30, 4, seed=30)
        ang_snap, _ = angles(U, V, alg="snap")
        ang_svd,  _ = angles(U, V, alg="svd")
        assert_allclose(ang_snap, ang_svd, atol=1e-10)

    def test_angles_calibration(self):
        U = _random_orthonormal(30, 4, seed=31)
        V = _random_orthonormal(30, 4, seed=32)
        _, calib = angles(U, V, compute_calibration=True)
        assert calib is not None
        assert calib.shape == (4, 4)
        assert_allclose(calib @ calib.T, np.eye(4), atol=1e-10)


class TestMinDistBase:

    def test_finds_correct_closest(self):
        """minDistBase returns the index of the closest basis."""
        U0 = _random_orthonormal(30, 4, seed=40)
        U1 = _random_orthonormal(30, 4, seed=41)
        U2 = _random_orthonormal(30, 4, seed=42)
        # measure basis = U0 perturbed slightly
        rng = np.random.default_rng(43)
        eps = 1e-4 * rng.standard_normal(U0.shape)
        Q, _ = np.linalg.qr(U0 + eps)
        measure = Q[:, :4]
        idx, dmin, dists, _ = minDistBase([U0, U1, U2], measure)
        assert idx == 0
        assert_allclose(dmin, dists[0])
        assert len(dists) == 3

    def test_returns_calibrations_when_requested(self):
        bases = [_random_orthonormal(20, 3, seed=i) for i in range(4)]
        measure = _random_orthonormal(20, 3, seed=99)
        _, _, _, calibs = minDistBase(bases, measure, compute_calibration=True)
        assert len(calibs) == 4


class TestRank1Update:

    def test_basis_remains_approximately_orthonormal(self):
        """After a rank-1 update the basis should remain nearly orthonormal."""
        N, k = 50, 5
        basis = _random_orthonormal(N, k, seed=50)
        # Construct new_vecs as a linear combination of basis columns so that
        # p = basis @ (basis.T @ new_vecs) is well away from zero
        rng = np.random.default_rng(51)
        new_vecs = basis @ rng.standard_normal((k, 1)) + 0.1 * rng.standard_normal((N, 1))
        rank1_update(basis, new_vecs)
        gram = basis.T @ basis
        assert_allclose(gram, np.eye(k), atol=1e-10)

    def test_fixed_stepsize(self):
        """rank1_update with explicit stepsize should not raise."""
        N, k = 30, 3
        basis = _random_orthonormal(N, k, seed=52)
        rng = np.random.default_rng(53)
        new_vecs = rng.standard_normal((N, 1))
        rank1_update(basis, new_vecs, stepsize=0.1)
        gram = basis.T @ basis
        assert_allclose(gram, np.eye(k), atol=1e-10)


class TestCnvxNnls:

    def test_nonnegative_solution(self):
        """All weights returned by cnvx_nnls should be >= 0."""
        rng = np.random.default_rng(60)
        k, p = 8, 5
        Z = rng.standard_normal((k, p))
        z = rng.standard_normal((k, 1))
        w = cnvx_nnls(z, Z)
        assert (w >= 0).all()

    def test_approximate_unit_sum(self):
        """With large ksi the weights should sum close to 1."""
        rng = np.random.default_rng(61)
        k, p = 8, 5
        Z = rng.standard_normal((k, p))
        z = rng.standard_normal((k, 1))
        w = cnvx_nnls(z, Z, ksi=1e8)
        assert_allclose(w.sum(), 1.0, atol=1e-4)

    def test_with_tikhonov_regularisation(self):
        """cnvx_nnls with mu param should still return non-negative weights."""
        rng = np.random.default_rng(62)
        k, p = 8, 5
        Z = rng.standard_normal((k, p))
        z = rng.standard_normal((k, 1))
        w = cnvx_nnls(z, Z, mu=1e-2)
        assert (w >= 0).all()


class TestDeterminePodAlg:

    def test_small_size_returns_snap(self):
        alg = _determine_pod_alg_square_matrices(None, 100)
        assert alg == "snap"

    def test_large_size_returns_svd(self):
        alg = _determine_pod_alg_square_matrices(None, 500)
        assert alg == "svd"

    def test_explicit_alg_passed_through(self):
        assert _determine_pod_alg_square_matrices("svd",  10) == "svd"
        assert _determine_pod_alg_square_matrices("snap", 10) == "snap"

    def test_invalid_alg_raises(self):
        with pytest.raises(AssertionError):
            _determine_pod_alg_square_matrices("bad", 10)

