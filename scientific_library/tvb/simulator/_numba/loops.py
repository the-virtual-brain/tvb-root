
from numba import cuda, float32

def make_loop(cfun, model, n_svar):
    "Construct CUDA device function for integration loop."

    @cuda.jit
    def loop(n_step, W, X, G):
        # TODO only for 1D grid/block dims
        t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ti = cuda.threadIdx.x
        x = cuda.shared.array((n_svar, 64), float32)
        g = G[t]
        for j in range(n_step):
            for i in range(W.shape[0]):
                x[:, ti] = X[i, :, t]
                model(x, g * cfun(W, X, i, t))
                X[i, :, t] = x[:, ti]

    # TODO hack
    loop.n_svar = n_svar
    return loop
