import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_orthonormal(n, k, seed=0):
    """Return an (n, k) matrix with orthonormal columns."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def _make_synthetic_data(n_state=30, n_ctrl=15, n_samples=40, n_params=3,
                          seed=0):
    """
    Build synthetic state-space / control-input datasets for p parameter points.

    Returns
    -------
    state_data       : list[ndarray], each (n_state, n_samples)
    prev_state_data  : list[ndarray], each (n_state, n_samples)
    ctrl_data        : list[ndarray], each (n_ctrl, n_samples)
    params           : ndarray  (1, n_params)  – 1-d parameter space
    """
    rng = np.random.default_rng(seed)
    params = np.linspace(0.5, 1.5, n_params)[None, :]   # shape (1, p)

    state_data, prev_state_data, ctrl_data = [], [], []
    for i in range(n_params):
        mu = params[0, i]
        # simple smooth synthetic trajectories that depend on mu
        t = np.linspace(0, 2 * np.pi, n_samples)
        x = np.linspace(0, 1, n_state)
        tt, xx = np.meshgrid(t, x)
        snap = mu * np.sin(xx * tt) + 0.1 * rng.standard_normal((n_state, n_samples))
        state_data.append(snap)
        prev_state_data.append(np.roll(snap, 1, axis=1))   # shift by one time step

        ctrl = mu * np.cos(np.linspace(0, np.pi, n_ctrl)[:, None] * t[None, :])
        ctrl_data.append(ctrl)

    return state_data, prev_state_data, ctrl_data, params
