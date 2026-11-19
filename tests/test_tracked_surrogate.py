"""
Tests for rom_am/tracked_fluid_surrogate.py
"""

import os
import sys
from types import SimpleNamespace
import pytest
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import patch, MagicMock
from helpers import _make_synthetic_data

from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog
import rom_am.tracked_fluid_surrogate as surrogate_module

# ── shared fixture ────────────────────────────────────────────────────────────

# Unseen parameter used for prediction initialization
_TEST_PARAMS = np.array([[0.75]])


@pytest.fixture(scope="module")
def trained_surrogate():
    """Train a TrackedFluidSurrog on synthetic data once for all tests."""
    n_state, n_ctrl, n_samples, n_params = 30, 15, 40, 3
    state_data, prev_state_data, ctrl_data, params = _make_synthetic_data(
        n_state=n_state, n_ctrl=n_ctrl, n_samples=n_samples,
        n_params=n_params, seed=100)

    surrog = TrackedFluidSurrog(
        maxLen=500,
        reTrainThres=500,
        updateBasis=False,
    )
    surrog.train(
        dispData=ctrl_data,
        fluidPrevData=prev_state_data,
        fluidData=state_data,
        params=params,
        rank_pres=3,
        rank_disp=2,
        kernel="thin_plate_spline",
        smoothing=1e-2,
        degree=1,
        norm=[True, True],
        center=[True, True],
        multiple_param_regressor=True,
        cleanup=False,
        alg="svd",
    )

    surrog.initialize_predictions(params=_TEST_PARAMS)
    return surrog, state_data, prev_state_data, ctrl_data, params


class TestTrackedFluidSurrogInit:

    def test_defaults(self):
        s = TrackedFluidSurrog()
        assert s.maxLen == 6900
        assert s.reTrainThres == 240
        assert s.updateBasis is False
        assert s._predictedBasis is False
        assert s.omega0 == 0.5
        assert len(s.trainIn) == 0

    def test_custom_params(self):
        s = TrackedFluidSurrog(maxLen=100, reTrainThres=10, updateBasis=True, m=3)
        assert s.maxLen == 100
        assert s.reTrainThres == 10
        assert s.updateBasis is True
        assert s._m == 3


class TestTrackedFluidSurrogSigmoid:

    def test_midpoint_is_half(self):
        # sigmoid(x=s) should equal 0.5 by definition.
        s_obj = TrackedFluidSurrog()
        n = 10
        val = s_obj.sigmoid(int(n / 2), n=n, eps=0.3)
        assert_allclose(val, 0.5, atol=1e-10)

    def test_monotone(self):
        s = TrackedFluidSurrog()
        n = 10
        vals = s.sigmoid(np.arange(n + 1), n=n, eps=0.3)
        assert np.all(np.diff(vals) >= 0), "sigmoid should be monotone increasing"

    def test_range(self):
        s = TrackedFluidSurrog()
        vals = s.sigmoid(np.linspace(-10, 10, 50), n=0, eps=1.)
        assert np.all(vals >= 0) and np.all(vals <= 1)


class TestTrackedFluidSurrogTrain:

    def test_train_sets_attributes(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        assert surrog._p == 3
        assert surrog.reducLoad is not None
        assert surrog.reducDisp is not None
        assert len(surrog.regressor) == 3

    def test_latent_dims_positive(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        assert surrog.reducLoad.latent_dim > 0
        assert surrog.reducDisp.latent_dim > 0

    def test_reduced_data_kept_when_cleanup_false(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        # cleanup=False was used
        assert surrog.reducedLoadData is not None
        assert len(surrog.reducedLoadData) == 3

    def test_params_stored(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        assert surrog.params.shape == (1, 3)


class TestTrackedFluidSurrogInitializePredictions:

    def test_initializes_once(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        assert surrog._predictedBasis

    def test_second_call_is_noop(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        surrog._predictedBasis = True
        surrog.initialize_predictions(params=_TEST_PARAMS)   # should not raise
        assert surrog._predictedBasis

    def test_calibration_matrices_shape(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        surrog._predictedBasis = False
        surrog.initialize_predictions(params=_TEST_PARAMS)
        r = surrog.reducLoad.latent_dim
        assert surrog.stacked_calib.shape == (3, r, r)


class TestTrackedFluidSurrogPredict:

    def test_output_shape_single_snapshot(self, trained_surrogate):
        surrog, state_data, _, ctrl_data, _ = trained_surrogate
        surrog._predictedBasis = False
        surrog.initialize_predictions(params=_TEST_PARAMS)
        # single time step
        ctrl_new  = ctrl_data[0][:, [0]]
        state_new = state_data[0][:, [0]]
        pred = surrog.predict(ctrl_new, state_new)
        assert pred.shape == (state_data[0].shape[0], 1)

    def test_output_shape_multiple_snapshots(self, trained_surrogate):
        surrog, state_data, prev_state_data, ctrl_data, params = trained_surrogate
        ctrl_new  = ctrl_data[0][:, :5]
        state_new = state_data[0][:, :5]
        pred = surrog.predict(ctrl_new, state_new, optimized=False)
        assert pred.shape == (state_data[0].shape[0], 5)

    def test_output_finite(self, trained_surrogate):
        surrog, state_data, _, ctrl_data, _ = trained_surrogate
        pred = surrog.predict(ctrl_data[1][:, [0]], state_data[1][:, [0]])
        assert np.isfinite(pred).all()

    def test_with_solid_reduc(self, trained_surrogate):
        """predict() also works when an external solidReduc is provided."""
        from rom_am.dimreducers.rom_am.podReducer import PodReducer
        surrog, state_data, prev_state_data, ctrl_data, params = trained_surrogate

        all_ctrl = np.hstack(ctrl_data)
        ext_reduc = PodReducer(latent_dim=2)
        ext_reduc.train(all_ctrl, normalize=True, center=True, alg="svd")

        ctrl_new  = ctrl_data[0][:, [0]]
        state_new = state_data[0][:, [0]]
        pred = surrog.predict(ctrl_new, state_new, solidReduc=ext_reduc)
        assert pred.shape[0] == state_data[0].shape[0]
        assert np.isfinite(pred).all()

    def test_takes_low_dimensional_disp(self, trained_surrogate):
        surrog, state_data, _, ctrl_data, _ = trained_surrogate
        r_ctrl = surrog.reducDisp.latent_dim
        ctrl_low = surrog.reducDisp.encode(ctrl_data[0][:, [0]], high_dim=True)
        pred = surrog.predict(ctrl_low, state_data[0][:, [0]], takes_low_dimensional_disp=True)
        assert pred.shape == (state_data[0].shape[0], 1)


class TestTrackedFluidSurrogComputeCalibrations:

    @pytest.fixture
    def calibration_case(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        surrog = TrackedFluidSurrog()
        matrices = [
            np.random.default_rng(42).standard_normal((4, 4)),
            np.diag([2., 1., 1e-12, 0.]),
            np.zeros((4, 4)),
        ]
        surrog.reducLoad = SimpleNamespace(pod=SimpleNamespace(modes=np.eye(4)))
        surrog.reducLoadLocals = [
            SimpleNamespace(pod=SimpleNamespace(modes=matrix)) for matrix in matrices
        ]
        expected = []
        for matrix in matrices:
            u, _, vh = surrogate_module.POD().decompose(matrix, alg="svd")
            expected.append(u @ vh)
        return surrog, np.stack(expected)

    @pytest.mark.skipif(surrogate_module._batched_calibrations is None, reason="Requires Linux JAX")
    @pytest.mark.parametrize("algorithm", [None, "svd"])
    def test_batched_svd_matches_pod(self, calibration_case, monkeypatch, algorithm):
        from threadpoolctl import threadpool_info

        surrog, expected = calibration_case
        monkeypatch.setenv("ROM_AM_SVD_THREADS", "2")
        original = surrogate_module._batched_calibrations

        def checked_batch(products):
            pools = [p for p in threadpool_info() if p["user_api"] == "blas"]
            assert pools and all(p["num_threads"] == 2 for p in pools)
            return original(products)

        monkeypatch.setattr(surrogate_module, "_batched_calibrations", checked_batch)
        with patch.object(surrogate_module, "Parallel") as parallel:
            before = threadpool_info()
            surrog._compute_calibrations(alg=algorithm)
            assert threadpool_info() == before
            parallel.assert_not_called()
        assert isinstance(surrog.calibrationQs, list)
        assert all(isinstance(q, np.ndarray) for q in surrog.calibrationQs)
        assert_allclose(surrog.stacked_calib, expected, atol=1e-12)
        assert_allclose(surrog.stacked_calib[1], np.diag([1., 1., 0., 0.]), atol=1e-12)
        assert_allclose(surrog.stacked_calib[2], np.zeros((4, 4)), atol=1e-12)

    @pytest.mark.parametrize("gil_enabled", [True, False, None])
    def test_no_jax_backend_selection(self, calibration_case, monkeypatch, gil_enabled):
        surrog, expected = calibration_case
        monkeypatch.setattr(surrogate_module, "_batched_calibrations", None)
        monkeypatch.setattr(surrogate_module.pod_module, "os_", 1)
        monkeypatch.setattr(surrogate_module.pod_module, "_physical_core_count", lambda: 2)
        if gil_enabled is None:
            monkeypatch.delattr(sys, "_is_gil_enabled", raising=False)
        else:
            monkeypatch.setattr(sys, "_is_gil_enabled", lambda: gil_enabled, raising=False)
        with patch.object(surrogate_module, "Parallel", wraps=surrogate_module.Parallel) as parallel, \
             patch.object(surrogate_module, "ThreadPoolExecutor",
                          wraps=surrogate_module.ThreadPoolExecutor) as executor:
            surrog._compute_calibrations(njobs=1, alg="svd")
        if gil_enabled is False:
            executor.assert_called_once_with(max_workers=2)
            parallel.assert_not_called()
        else:
            parallel.assert_called_once_with(n_jobs=1, backend="loky", prefer="threads")
            executor.assert_not_called()
        assert_allclose(surrog.stacked_calib, expected, atol=1e-12)

    def test_explicit_snapshot_bypasses_batch(self, calibration_case):
        surrog, _ = calibration_case
        with patch.object(surrogate_module, "_batched_calibrations") as batch:
            surrog._compute_calibrations(alg="snap")
            batch.assert_not_called()
        expected = []
        for local in surrog.reducLoadLocals:
            u, _, vh = surrogate_module.POD().decompose(local.pod.modes, alg="snap")
            expected.append(u @ vh)
        assert_allclose(surrog.stacked_calib, expected, atol=1e-12)

    def test_precomputed_calibrations_bypass_backends(self, calibration_case):
        surrog, expected = calibration_case
        precomputed = list(expected)
        with patch.object(surrogate_module, "_batched_calibrations") as batch, \
             patch.object(surrogate_module, "Parallel") as parallel, \
             patch.object(surrogate_module, "ThreadPoolExecutor") as executor:
            surrog._compute_calibrations(already_computed_calibs=precomputed)
            batch.assert_not_called()
            parallel.assert_not_called()
            executor.assert_not_called()
        assert surrog.calibrationQs is precomputed
        assert_allclose(surrog.stacked_calib, expected)

    def test_calibration_unitary(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        surrog._predictedBasis = False
        surrog.initialize_predictions(params=_TEST_PARAMS)
        r = surrog.reducLoad.latent_dim
        for Q in surrog.calibrationQs:
            assert Q.shape == (r, r)
            assert_allclose(Q @ Q.T, np.eye(r), atol=1e-10)

    def test_stacked_calib_shape(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        r  = surrog.reducLoad.latent_dim
        p  = len(surrog.reducLoadLocals)
        assert surrog.stacked_calib.shape == (p, r, r)


class TestTrackedFluidSurrogReTrain:

    def test_retrain_creates_new_regressor(self, trained_surrogate):
        surrog, _, _, _, _ = trained_surrogate
        # put some data into the deque manually
        surrog._predictedBasis = False
        surrog.initialize_predictions(params=_TEST_PARAMS)

        r_ctrl  = surrog.reducDisp.latent_dim
        r_state = surrog.reducLoad.latent_dim
        rng = np.random.default_rng(200)
        for _ in range(20):
            inp = rng.standard_normal(r_ctrl + r_state)
            out = rng.standard_normal(r_state)
            surrog.trainIn.appendleft(inp)
            surrog.trainOut.appendleft(out)

        surrog.new_regressor = None
        with patch("builtins.open", MagicMock()), \
             patch("numpy.save",    MagicMock()):
            surrog._reTrain(weights=False)
        assert surrog.new_regressor is not None

    def test_retrain_with_weights(self, trained_surrogate):
        surrog, _, _, _, _ = trained_surrogate
        surrog._predictedBasis = False
        surrog.initialize_predictions(params=_TEST_PARAMS)

        r_ctrl  = surrog.reducDisp.latent_dim
        r_state = surrog.reducLoad.latent_dim
        rng = np.random.default_rng(201)
        for _ in range(20):
            inp = rng.standard_normal(r_ctrl + r_state)
            out = rng.standard_normal(r_state)
            surrog.trainIn.appendleft(inp)
            surrog.trainOut.appendleft(out)

        surrog.new_regressor = None
        with patch("builtins.open", MagicMock()), \
             patch("numpy.save",    MagicMock()):
            surrog._reTrain(weights=True)
        assert surrog.new_regressor is not None

    def test_retrain_prediction_still_works(self, trained_surrogate):
        surrog, state_data, prev_state_data, ctrl_data, params = trained_surrogate
        pred = surrog.predict(ctrl_data[0][:, [0]], state_data[0][:, [0]])
        assert np.isfinite(pred).all()


class TestTrackedFluidSurrogAugmentData:

    def _fresh_surrogate(self):
        n_state, n_ctrl, n_samples, n_params = 30, 15, 40, 3
        state_data, prev_state_data, ctrl_data, params = _make_synthetic_data(
            n_state=n_state, n_ctrl=n_ctrl, n_samples=n_samples,
            n_params=n_params, seed=300)

        surrog = TrackedFluidSurrog(
            maxLen=500,
            reTrainThres=500,
            updateBasis=False,
        )
        surrog.train(
            dispData=ctrl_data,
            fluidPrevData=prev_state_data,
            fluidData=state_data,
            params=params,
            rank_pres=3,
            rank_disp=2,
            kernel="thin_plate_spline",
            smoothing=1e-2,
            degree=1,
            multiple_param_regressor=True,
            cleanup=False,
            alg="svd",
        )
        surrog.initialize_predictions(params=_TEST_PARAMS)
        return surrog, state_data, prev_state_data, ctrl_data, params

    def test_deque_grows(self):
        surrog, state_data, prev_state_data, ctrl_data, _ = self._fresh_surrogate()
        initial_len = len(surrog.trainIn)
        with patch("builtins.open", MagicMock()), \
             patch("numpy.save",    MagicMock()):
            surrog.augmentData(
                ctrl_data[0][:, [0]],
                prev_state_data[0][:, [0]],
                state_data[0][:, [0]],
            )
        assert len(surrog.trainIn) == initial_len + 1

    def test_count_augment_increments(self):
        surrog, state_data, prev_state_data, ctrl_data, _ = self._fresh_surrogate()
        initial_count = surrog.countAugment
        with patch("builtins.open", MagicMock()), \
             patch("numpy.save",    MagicMock()):
            surrog.augmentData(
                ctrl_data[0][:, [0]],
                prev_state_data[0][:, [0]],
                state_data[0][:, [0]],
            )
        assert surrog.countAugment == initial_count + 1

    def test_retrain_triggered_at_threshold(self):
        """When countAugment exceeds reTrainThres, _reTrain is called."""
        surrog, state_data, prev_state_data, ctrl_data, _ = self._fresh_surrogate()
        # The RBF interpolator needs >= 6 pts.
        r_ctrl  = surrog.reducDisp.latent_dim
        r_state = surrog.reducLoad.latent_dim
        rng = np.random.default_rng(700)
        for _ in range(20):
            inp = rng.standard_normal(r_ctrl + r_state)
            out = rng.standard_normal(r_state)
            surrog.trainIn.appendleft(inp)
            surrog.trainOut.appendleft(out)
        surrog.reTrainThres = 3
        surrog.countAugment = 3   # next call will exceed threshold

        with patch("builtins.open", MagicMock()), \
             patch("numpy.save",    MagicMock()):
            surrog.augmentData(
                ctrl_data[0][:, [0]],
                prev_state_data[0][:, [0]],
                state_data[0][:, [0]],
            )
        # after retrain countAugment is reset to 0
        assert surrog.countAugment == 0
        assert surrog.retrain_count == 1

    def test_send_signal_basis_none_by_default(self):
        surrog, state_data, prev_state_data, ctrl_data, _ = self._fresh_surrogate()
        with patch("builtins.open", MagicMock()), \
             patch("numpy.save",    MagicMock()):
            surrog.augmentData(
                ctrl_data[0][:, [0]],
                prev_state_data[0][:, [0]],
                state_data[0][:, [0]],
            )
        assert surrog.sendSignalBasis is None


class TestTrackedFluidSurrogDetermineAutomaticOmega:

    def test_omega_in_unit_interval(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        surrog.number_of_initial_snaps = 40
        r_ctrl  = surrog.reducDisp.latent_dim
        r_state = surrog.reducLoad.latent_dim
        rng = np.random.default_rng(400)
        for _ in range(10):
            inp = rng.standard_normal(r_ctrl + r_state)
            out = rng.standard_normal(r_state)
            surrog.trainIn.appendleft(inp)
            surrog.trainOut.appendleft(out)
        surrog._determine_automatic_omega()
        assert 0.0 <= surrog.omega0 <= 1.0

    def test_omega_terms_sum_to_one(self, trained_surrogate):
        surrog, *_ = trained_surrogate
        surrog._determine_automatic_omega()
        assert_allclose(sum(surrog._omega_terms), 1.0, atol=1e-12)


class TestTrackedFluidSurrogSave:

    def test_save_creates_pkl(self, tmp_path, trained_surrogate):
        surrog, *_ = trained_surrogate
        fpath = str(tmp_path / "surrogate")
        surrog.save(fpath)
        assert os.path.exists(fpath + ".pkl")

    def test_save_load_roundtrip(self, tmp_path, trained_surrogate):
        import pickle
        surrog, state_data, prev_state_data, ctrl_data, params = trained_surrogate
        fpath = str(tmp_path / "surrogate_rt")
        surrog.save(fpath)
        with open(fpath + ".pkl", "rb") as f:
            surrog2 = pickle.load(f)
        pred1 = surrog.predict( ctrl_data[0][:, [0]], state_data[0][:, [0]])
        pred2 = surrog2.predict(ctrl_data[0][:, [0]], state_data[0][:, [0]])
        assert_allclose(pred1, pred2, atol=1e-12)
