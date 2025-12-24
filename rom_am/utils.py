from scipy.linalg import logm, expm
import numpy as np
from rom_am.pod import POD
from scipy.optimize import nnls
from joblib import Parallel, delayed

def log_A(A, At, exp = False):
    if exp:
        return logm(At @ np.linalg.inv(A))
    else:
        return At - A

def exp_A(A, At, exp = False):
    if exp:
        return expm(At) @ A
    else:
        return A + At

def log_U(U, Ut) -> np.array:

    N = U.shape[0]

    prod = POD()
    u, _, rh = prod.decompose(Ut.T @ U,)
    procr = np.linalg.multi_dot((Ut, u, rh))
    L = (np.eye(N) - U @ U.T) @ (procr)

    prod2 = POD()
    q, sig, vh = prod2.decompose(L, thin=True)

    return np.linalg.multi_dot((q, np.diag(np.arcsin(sig)), vh))

def exp_U(U, delt):

    prod = POD()
    q, sig, vh = prod.decompose(delt, thin=True)

    return np.linalg.multi_dot((U, vh.T, np.diag(np.cos(sig)), vh)) + np.linalg.multi_dot((q, np.diag(np.sin(sig)), vh))

def angles(U, V, alg=None):
    alg = _determine_pod_alg_square_matrices(alg, U.shape[1])
    prod = POD()
    _, sig, _ = prod.decompose(U.T @ V, thin=True, alg=alg)

    return np.arccos(sig[sig<1])

def dist(U, V, alg=None):
    return np.sqrt(np.sum(angles(U, V, alg=alg)**2))


def minDistBase(bases_list, measureBase, njobs=1, alg=None):

    dists = np.array(Parallel(n_jobs=njobs, backend='loky', prefer='threads')(
        delayed(dist)(base, measureBase, alg) for base in bases_list
    ))
    id_ = np.argmin(dists)
    return id_, dists[id_], dists

def rank1_update(basis:np.ndarray, new_vectors: np.ndarray, stepsize = None):
    """Updating the basis matrix using GROUSE 1 rank update

    Parameters
    ----------
    base        : numpy.ndarray
                 Current basis matrix, of shape (n, r)
    new_vectors : numpy.ndarray
                 New observed vectors, of shape (n, m)
    Returns
    ----------

    """

    w = basis.T @ new_vectors
    p = basis @ w
    r = new_vectors - p
    r_norm = np.linalg.norm(r)
    p_norm = np.linalg.norm(p)

    if stepsize is None:
        stepsize_ = np.arcsin(r_norm/p_norm)
    else:
        stepsize_ = stepsize * (r_norm * p_norm)

    w_norm = np.linalg.norm(w)
    coeff1 = (np.cos(stepsize_) - 1) / p_norm
    coeff2 = np.sin(stepsize_) / r_norm
    basis += (coeff1 * p + coeff2 * r) @ (w.T / w_norm)

def cnvx_nnls(z, Z, ksi=1e5, mu=None):
    """Solving a non negative linear square problem
    min || Z w - z ||
    with the constraint of w_i > 1 for all i
    This uses scipy.optimize 's nnls function

    Parameters
    ----------
    z : numpy.ndarray
        Right hand vector, of shape (k, 1)
    Z : numpy.ndarray
        Matrix Z as shown above, of shape (k, p)
    ksi: float
        Value of Penalization parameter ksi_ to enforce \Sum w_i = 1
        where ksi_ = ksi * tr(Z.T Z)/dim(w)
        Default : 1e5
    mu: None or float
        Value of Tikhonov regularization parameter, thus solving
        min || Z w - z || + lambda || w ||
        where lambda = mu * tr(Z.T Z)/dim(w)
        Default : None
    Returns
    ----------
    w : numpy.ndarray, size (p, )
        solution of the constrained nnls problem

    """

    k = Z.shape[1]
    ksi_ = ksi * np.trace(Z.T @ Z)/k

    Zaug = np.vstack((Z, np.sqrt(ksi_) * np.ones((1, k))))
    zaug = np.vstack((z, np.array([[np.sqrt(ksi_)]])))

    if mu is not None:
        mu_ = mu * np.trace(Zaug.T @ Zaug)/k
        Zaug = np.vstack((Zaug, mu_ * np.eye(k)))
        zaug = np.vstack((zaug, np.zeros((k, 1))))

    w, _ = nnls(Zaug, zaug.ravel())

    return w

def _determine_pod_alg_square_matrices(alg, size_):
    if alg is None:
        if size_ < 400:
            alg = "snap"
        else:
            alg = "svd"
    else:
        assert (alg == "svd" or alg == "snap"), "The algorithm chosen must be `snap` or `svd`"
    return alg
