import numpy as np
from rom_am.dmdc import DMDc


# DMD with control
u = np.ones((1, 4))
X = np.array([[4.,  3.,  1.5, -0.75],
              [7.,  0.7,  0.07,  0.007]])
Y = np.array([[3.000e+00,  1.500e+00, -7.500e-01, -4.125e+00],
              [7.000e-01,  7.000e-02,  7.000e-03,  7.000e-04]])
A = np.array([[1.5, 0], [0, 0.1]])
B = np.array([-3, 0]).reshape((-1, 1))

dt = 1
dmdc = DMDc()
dmdc.decompose(X,  Y=Y, dt=dt, Y_input=u)


def test_dmdc_decomposition():
    assert dmdc.dmd_modes is not None and dmdc.A_tilde is not None and dmdc.B_tilde is not None


def test_dmdc_operators():
    op1 = np.allclose(dmdc.u_hat @ dmdc.A_tilde @ dmdc.u_hat.T, A)
    op2 = np.allclose(dmdc.u_hat @ dmdc.B_tilde, B)
    assert op1 and op2


def test_dmdc_reconstruction():
    assert np.allclose(dmdc.u_hat @ dmdc.A_tilde @ dmdc.u_hat.T @
                       X + dmdc.u_hat @ dmdc.B_tilde @ u, Y)


# ----------------- Testing Discrete prediction ---------------------
X1 = np.zeros((2, 21))
Y1 = np.zeros((2, 21))
u1 = np.ones((1, 21))
X1[:, 0] = np.array([4, 7])
Y1[:, 0] = A @ X1[:, 0] + B.ravel() * u1[:, 0]
for i in range(Y1.shape[1]-1):
    Y1[:, i+1] = A @ Y1[:, i] + B.ravel() * u1[:, i]
X1[:, 1::] = Y1[:, :-1]


def test_dmdc_discr_pred():
    dt = 10
    t = np.arange(0, 200, dt)
    predicted_X = np.real(dmdc.predict(
        t, t1=t[0], u_input=np.ones((1, t.shape[0]))))
    assert np.allclose(X1, predicted_X)
