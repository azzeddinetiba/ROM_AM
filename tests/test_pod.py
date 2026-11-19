import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

import rom_am.pod as pod_module
from rom_am.pod import POD


@pytest.mark.parametrize("alg", ["svd", "snap"])
def test_pod_decomposition_and_reconstruction(alg):
    data = np.array([[3., 0., 0.], [4., 0., 0.], [0., 0., 2.], [0., 0., 0.]])
    model = POD()
    u, s, vh = model.decompose(data, alg=alg)

    assert model.kept_rank == 2
    assert u.shape == (4, 2) and vh.shape == (2, 3)
    np.testing.assert_allclose(s, [5., 2.], atol=1e-12)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(model.reconstruct(), data, atol=1e-12)


@pytest.mark.parametrize("alg", ["svd", "snap"])
def test_pod_rank_truncation(alg):
    model = POD()
    model.decompose(np.diag([5., 2., 1.]), alg=alg, rank=2)

    assert model.kept_rank == 2
    np.testing.assert_allclose(model.singvals, [5., 2.], atol=1e-12)
    np.testing.assert_allclose(model.reconstruct(), np.diag([5., 2., 0.]), atol=1e-12)


@pytest.mark.skipif("linux" not in sys.platform, reason="Linux SVD threading")
@pytest.mark.parametrize(
    "physical,available,expected", [(6, 12, 6), (6, 2, 2), (6, 6, 6)]
)
def test_svd_physical_core_default(monkeypatch, physical, available, expected):
    monkeypatch.delenv("ROM_AM_SVD_THREADS", raising=False)
    monkeypatch.setattr(pod_module.psutil, "cpu_count", lambda logical: physical)
    monkeypatch.setattr(pod_module.os, "sched_getaffinity", lambda pid: set(range(available)))
    assert pod_module._svd_thread_count() == expected


@pytest.mark.skipif("linux" not in sys.platform, reason="Linux SVD threading")
def test_svd_unknown_physical_cores(monkeypatch):
    monkeypatch.delenv("ROM_AM_SVD_THREADS", raising=False)
    monkeypatch.setattr(pod_module.psutil, "cpu_count", lambda logical: None)
    with pytest.warns(RuntimeWarning, match="Could not detect physical CPU cores"):
        assert pod_module._svd_thread_count() == 1


def test_physical_cores_without_linux_affinity(monkeypatch):
    monkeypatch.delattr(pod_module.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(pod_module.psutil, "cpu_count", lambda logical: 6)
    monkeypatch.setattr(pod_module.os, "cpu_count", lambda: 4)
    assert pod_module._physical_core_count() == 4


@pytest.mark.skipif("linux" not in sys.platform, reason="Linux SVD threading")
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid", ""])
def test_svd_invalid_thread_override(monkeypatch, value):
    monkeypatch.setenv("ROM_AM_SVD_THREADS", value)
    with pytest.raises(ValueError, match="ROM_AM_SVD_THREADS must be a positive integer"):
        POD().decompose(np.eye(3))


@pytest.mark.skipif("linux" not in sys.platform, reason="Linux SVD threading")
def test_svd_thread_limits_and_reconstruction(monkeypatch):
    from threadpoolctl import threadpool_info

    monkeypatch.setenv("ROM_AM_SVD_THREADS", "2")
    svd = pod_module.jnp.linalg.svd
    observed = []

    def checked_svd(*args, **kwargs):
        pools = [p for p in threadpool_info() if p["user_api"] == "blas"]
        assert pools
        assert all(p["num_threads"] == 2 for p in pools)
        observed.append(True)
        return svd(*args, **kwargs)

    monkeypatch.setattr(pod_module.jnp.linalg, "svd", checked_svd)
    before = threadpool_info()
    data = np.random.default_rng(42).standard_normal((30, 8))
    barrier = Barrier(2)

    def decompose():
        barrier.wait()
        model = POD()
        u, s, vh = model.decompose(data)
        assert all(isinstance(a, np.ndarray) for a in (u, s, vh))
        assert u.shape == (30, 8) and s.shape == (8,) and vh.shape == (8, 8)
        np.testing.assert_allclose(s, np.linalg.svd(data, compute_uv=False), rtol=1e-12)
        np.testing.assert_allclose(model.reconstruct(), data, atol=1e-12)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: decompose(), range(2)))
    assert len(observed) == 2
    assert threadpool_info() == before


@pytest.mark.skipif("linux" not in sys.platform, reason="Linux SVD threading")
def test_svd_restores_limits_on_failure(monkeypatch):
    from threadpoolctl import threadpool_info

    monkeypatch.setenv("ROM_AM_SVD_THREADS", "2")

    def failing_svd(*args, **kwargs):
        raise RuntimeError("SVD failed")

    monkeypatch.setattr(pod_module.jnp.linalg, "svd", failing_svd)
    before = threadpool_info()
    with pytest.raises(RuntimeError, match="SVD failed"):
        POD().decompose(np.eye(3))
    assert threadpool_info() == before


@pytest.mark.parametrize("algorithm,platform", [("snap", 0), ("svd", 1)])
def test_other_paths_ignore_linux_thread_override(monkeypatch, algorithm, platform):
    monkeypatch.setenv("ROM_AM_SVD_THREADS", "invalid")
    monkeypatch.setattr(pod_module, "os_", platform)
    model = POD()
    model.decompose(np.eye(3), alg=algorithm)
    np.testing.assert_allclose(model.reconstruct(), np.eye(3), atol=1e-12)


@pytest.mark.skipif("linux" not in sys.platform, reason="Linux optional JAX")
def test_missing_jax_uses_scipy():
    code = """
import builtins
import warnings
from types import SimpleNamespace
original_import = builtins.__import__
def without_jax(name, *args, **kwargs):
    if name == "jax" or name.startswith("jax."):
        raise ModuleNotFoundError("No module named 'jax'", name="jax")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_jax
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import rom_am.pod as pod
assert pod.jax is None and pod.os_ == 1
assert any("JAX is unavailable" in str(w.message) for w in caught)
model = pod.POD()
model.decompose(pod.np.eye(3))
pod.np.testing.assert_allclose(model.reconstruct(), pod.np.eye(3))
from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog, _batched_calibrations
assert _batched_calibrations is None
surrog = TrackedFluidSurrog.__new__(TrackedFluidSurrog)
surrog.reducLoad = SimpleNamespace(pod=SimpleNamespace(modes=pod.np.eye(3)))
surrog.reducLoadLocals = [surrog.reducLoad]
surrog._compute_calibrations(alg="svd")
pod.np.testing.assert_allclose(surrog.stacked_calib, [pod.np.eye(3)])
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
