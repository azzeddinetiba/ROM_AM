import os
import numpy as np
from rom_am.edmd import EDMD

"""
Tests on Extended DMD based on the nonlinear spring example (with unsteady
pressure load) studied in the eDMD_Spring example
"""
input_t = np.load("./tests/DATA/edmd_input_t.npy")
train_t = input_t[:int(0.7*input_t.shape[0])]
input_data = np.load("./tests/DATA/edmd_input_data.npy")
dt = 6e-4

mass = 1000
rigid = 1e7
mu_coeff = 6
mu = mu_coeff * rigid / 0.2
A = 1
pres_init0 = 1e5
Ls0 = 1.2
L0 = 1
pres = 1e6
def p(t): return pres * np.sin(2 * np.pi * 20 * t)


a_ = rigid
b_ = mu
c_ = A * pres_init0
interm = (((np.sqrt((27 * b_ * c_**2 + 4 * a_**3) / b_)) /
           (b_ * 2 * 3**(3. / 2.))) - c_ / (2 * b_))**(1. / 3.)
u0 = interm - a_ / (3 * b_ * interm)

a = -rigid/mass - 3*mu*(u0**2)/mass
b = 3*mu*u0/mass
c = -mu/mass
d1 = (u0**3)*mu/mass + rigid * u0/mass
d2 = A/mass
def d(t): return d1 + d2 * p(t)
def model_param(t): return np.array([a, b, c, d(t), d1, d2])
def v_dot(u, t): return c * u**3 + b * u**2 + a * u + d(t)


correct_model = model_param(0)[[0, 1, 2, 4, 5]]

X = np.load("./tests/DATA/edmd_train_X.npy")
Y = np.load("./tests/DATA/edmd_train_Y.npy")
observables = {"X": [lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: np.ones((1, X.shape[1])), lambda x: p(train_t).reshape((1, -1))],
               "Y": [lambda x: v_dot(x, train_t)]}

edmd = EDMD()
edmd.decompose(X,  Y=Y, dt=dt, observables=observables)


def test_edmd_decomposition():
    assert edmd.modes is not None and edmd.A is not None and np.allclose(
        edmd.A, model_param(0)[[0, 1, 2, 4, 5]])


def test_edmd_prediction():
    pred_t = input_t.copy()
    input_ = np.vstack((input_data[0, :], input_data[0, :]**2, input_data[0, :]
                        ** 3, np.ones((1, len(pred_t))), p(input_t).reshape((1, -1))))
    pred = np.real(edmd.predict(pred_t, x_input=input_))
    correct_sol = v_dot(input_data[0, :], input_t)
    predicted_sol = pred[0, :]
    assert np.allclose(correct_sol, predicted_sol)
