import numpy as np
import numba as nb

@nb.njit(fastmath=True, boundscheck=False, parallel=True)
def delays(dt, r, V, weights,idelays,g,Delta,tau,eta,J,I):
    rtau = nb.float32(1 / tau)
    Delta_rpitau = nb.float32(Delta / (np.pi * tau))
    for k in nb.prange(r.shape[0]):
        for t in range(${nt}):
            for i in range(r.shape[1]):
% for i in range(nl):
                acc${i} = nb.float32(0.0)
% endfor
                for j in range(r.shape[1]):
% for i in range(nl):
                    acc${i} += weights[i,j]*r[k, j, ${nh} + t - idelays[i, j], ${i}]
% endfor
% for i in range(nl):
                r_c${i} = g[k,${i}] * acc${i}
                r${i} = r[k, i, ${nh} + t, ${i}]
                V${i} = V[k, i, ${nh} + t, ${i}]
                r_noise${i} = r[k, i, ${nh} + t + 1, ${i}]
                V_noise${i} = V[k, i, ${nh} + t + 1, ${i}]
                dr${i} = rtau * (Delta_rpitau + 2 * V${i} * r${i})
                dV${i} = 1/tau * ( V${i}**2 - np.pi**2 * tau**2 * r${i}**2 + eta + J * tau * r${i} + I + r_c${i} ) 
                r[k, i, ${nh} + t + 1, ${i}] = r${i} + dt*dr${i} + r_noise${i}
                V[k, i, ${nh} + t + 1, ${i}] = V${i} + dt*dV${i} + V_noise${i}
% endfor

