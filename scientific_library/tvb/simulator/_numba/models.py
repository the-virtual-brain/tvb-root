
import numpy
from numba import cuda, float32, guvectorize, float64


def make_bistable():
    "Construct CUDA device function for a bistable system."

    @cuda.jit(device=True)
    def f(dX, X, I):
        t = cuda.threadIdx.x
        x = X[0, t]
        dX[0, t] = (x - x*x*x - float32(1.0) + I) / float32(50.0)

    return f


def make_jr():
    "Construct CUDA device function for the Jansen-Rit model."

    # parameters
    A   ,    B,   a,    b,   v0, nu_max,    r,     J, a_1, a_2,  a_3,  a_4, p_min, p_max,   mu = map(float32, [
    3.25, 22.0, 0.1, 0.05, 5.52, 0.0025, 0.56, 135.0, 1.0, 0.8, 0.25, 0.25,  0.12,  0.32, 0.22])

    # jit maps this to CUDA exp
    from math import exp

    @cuda.jit(device=True)
    def f(dX, X, I):
        one, two = float32(1.0), float32(2.0)
        t = cuda.threadIdx.x
        dX[0, t] = X[3, t]
        dX[1, t] = X[4, t]
        dX[2, t] = X[5, t]
        dX[3, t] = A * a * two * nu_max / (one + exp(r * (v0 - (X[1, t] - X[2, t])))) - float32(2.0) * a * X[3, t] - a * a * X[0, t]
        dX[4, t] = A * a * (mu + a_2 * J * two * nu_max / (one + exp(r * (v0 - (a_1 * J * X[0, t])))) + I) - two * a * X[4, t] - a * a * X[1, t]
        dX[5, t] = B * b * (a_4 * J * two * nu_max / (one + exp(r * (v0 - (a_3 * J * X[0, t]))))) - two * b * X[5, t] - b * b * X[2, t]

    return f


@guvectorize([(float64[:],)*11], '(n),(m)' + ',()'*8 + '->(n)', nopython=True)
def rww_dfun(S, c, a, b, d, g, ts, w, j, io, dx):
    "Gufunc for reduced Wong-Wang model equations."

    if S[0] < 0.0:
        dx[0] = 0.0 - S[0]
    elif S[0] > 1.0:
        dx[0] = 1.0 - S[0]
    else:
        x = w[0]*j[0]*S[0] + io[0] + j[0]*c[0]
        h = (a[0]*x - b[0]) / (1 - numpy.exp(-d[0]*(a[0]*x - b[0])))
        dx[0] = - (S[0] / ts[0]) + (1.0 - S[0]) * h * g[0]


@guvectorize([(float64[:],) * 18], '(n),(m)' + ',()'*15 + '->(n)', nopython=True)
def hmje_dfun(y, c_pop, x0, Iext, Iext2, a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, tau, ydot):
    "Gufunc for Hindmarsh-Rose-Jirsa Epileptor model equations."

    c_pop1 = c_pop[0]
    c_pop2 = c_pop[1]

    # population 1
    if y[0] < 0.0:
        ydot[0] = - a[0] * y[0] ** 2 + b[0] * y[0]
    else:
        ydot[0] = slope[0] - y[3] + 0.6 * (y[2] - 4.0) ** 2
    ydot[0] = tt[0] * (y[1] - y[2] + Iext[0] + Kvf[0] * c_pop1 + ydot[0] * y[0])
    ydot[1] = tt[0] * (c[0] - d[0] * y[0] ** 2 - y[1])

    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    ydot[2] = tt[0] * (r[0] * (4 * (y[0] - x0[0]) - y[2] + ydot[2] + Ks[0] * c_pop1))

    # population 2
    ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + 2 * y[5] - 0.3 * (y[2] - 3.5) + Kf[0] * c_pop2)
    if y[3] < -0.25:
        ydot[4] = 0.0
    else:
        ydot[4] = aa[0] * (y[3] + 0.25)
    ydot[4] = tt[0] * ((-y[4] + ydot[4]) / tau[0])

    # filter
    ydot[5] = tt[0] * (-0.01 * (y[5] - 0.1 * y[0]))
