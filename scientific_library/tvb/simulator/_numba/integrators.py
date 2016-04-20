
from numba import cuda, int32, float32

def make_euler(dt, f, n_svar, n_step):
    "Construct CUDA device function for Euler scheme."

    n_step = int32(n_step)
    dt = float32(dt)

    @cuda.jit(device=True)
    def scheme(X, I):
        dX = cuda.local.array((n_svar,), float32)
        for i in range(n_step):
            f(dX, X, I)
            for j in range(n_svar):
                X[j] += dX[j]

    return scheme


def make_rk4(dt, f, n_svar, n_step):
    "Construct CUDA device function for Runge-Kutta 4th order scheme."

    n_step = int32(n_step)
    dt = float32(dt)

    @cuda.jit(device=True)
    def scheme(X, I):
        k = cuda.shared.array((4, n_svar, 64), float32)
        x = cuda.shared.array((n_svar, 64), float32)
        t = cuda.threadIdx.x
        for i in range(n_step):
            f(k[0], X, I)
            for j in range(n_svar):
                x[j, t] = X[j, t] + (dt / float32(2.0)) * k[0, j, t]
            f(k[1], x, I)
            for j in range(n_svar):
                x[j, t] = X[j, t] + (dt / float32(2.0)) * k[1, j, t]
            f(k[2], x, I)
            for j in range(n_svar):
                x[j, t] = X[j, t] + dt * k[2, j, t]
            f(k[3], x, I)
            for j in range(n_svar):
                X[j, t] += (dt/float32(6.0)) * (k[0, j, t] + k[3, j, t] + float32(2.0)*(k[1, j, t] + k[2, j, t]))

    return scheme