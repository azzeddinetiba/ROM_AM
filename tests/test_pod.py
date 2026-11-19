import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

import rom_am.pod as pod_module
from rom_am.pod import POD


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
