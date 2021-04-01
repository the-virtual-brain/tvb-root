import numpy as np
import numba
from tvb.simulator.lab import *


def run_sim(sim, nstep):
    horizon = sim.connectivity.horizon
    buf_len = horizon + nstep
    N = sim.connectivity.number_of_regions
    gf = sim.integrator.noise.gfun(None)

    r, V = sim.integrator.noise.generate( shape=(2,N,buf_len) ) * gf
    r[:,:horizon] = np.roll(sim.history.buffer[:,0,:,0], -1, axis=0).T
    V[:,:horizon] = np.roll(sim.history.buffer[:,1,:,0], -1, axis=0).T


    r, V = _mpr_integrate(
        N = N,
        dt = sim.integrator.dt,
        nstep = nstep,
        i0 = horizon,
        r=r,
        V=V,
        weights = sim.connectivity.weights, 
        idelays = sim.connectivity.idelays,
        G = sim.coupling.a.item(),
        I = sim.model.I.item(),
        Delta = sim.model.Delta.item(), 
        Gamma = sim.model.Gamma.item(),
        eta = sim.model.eta.item(),
        tau = sim.model.tau.item(),
        J = sim.model.J.item(),       # end of model params
    )
    return r, V
        

def run_sim_tavg_chunked(sim, nstep, chunksize=100000):
    # chunksize in number of steps 
    horizon = sim.connectivity.horizon
    N = sim.connectivity.number_of_regions
    gf = sim.integrator.noise.gfun(None)
    assert len(sim.monitors) == 1
    assert isinstance(sim.monitors[0], monitors.TemporalAverage)

    tavg_steps=sim.monitors[0].istep
    assert tavg_steps < chunksize
    assert chunksize % tavg_steps == 0
    tavg_chunksize = chunksize // tavg_steps


    assert nstep % tavg_steps == 0
    r_out, V_out = np.zeros((2,N,nstep//tavg_steps))

    r, V = sim.integrator.noise.generate( shape=(2,N,chunksize+horizon) ) * gf
    r[:,:horizon] = np.roll(sim.history.buffer[:,0,:,0], -1, axis=0).T
    V[:,:horizon] = np.roll(sim.history.buffer[:,1,:,0], -1, axis=0).T

    for chunk, _ in enumerate(range(horizon, nstep+horizon, chunksize)):
        r, V = _mpr_integrate(
            N = N,
            dt = sim.integrator.dt,
            nstep = chunksize,
            i0 = horizon,
            r=r,
            V=V,
            weights = sim.connectivity.weights, 
            idelays = sim.connectivity.idelays,
            G = sim.coupling.a.item(),
            I = sim.model.I.item(),
            Delta = sim.model.Delta.item(), 
            Gamma = sim.model.Gamma.item(),
            eta = sim.model.eta.item(),
            tau = sim.model.tau.item(),
            J = sim.model.J.item(),       # end of model params
        )

        tavg_chunk = chunk * tavg_chunksize
        r_out[:,tavg_chunk:tavg_chunk+tavg_chunksize] = _time_average(r[:, horizon:], tavg_steps)
        V_out[:,tavg_chunk:tavg_chunk+tavg_chunksize] = _time_average(V[:, horizon:], tavg_steps)

        r[:,:horizon] = r[:,-horizon:]
        V[:,:horizon] = V[:,-horizon:]

        r_noise, V_noise = sim.integrator.noise.generate( shape=(2,N,chunksize) ) * gf
        r[:,horizon:] = r_noise
        V[:,horizon:] = V_noise

    return r_out, V_out

def _time_average(ts, istep):
    N, T = ts.shape
    return np.mean(ts.reshape(N,T//istep,istep),-1) # length of ts better be multiple of istep 


@numba.njit
def _mpr_integrate(
        N,       # number of regions
        dt,
        nstep,   # integration length
        i0,      # index to t0
        r,       # r buffer with initial history and pre-filled with noise
        V,       # V buffer with initial history and pre-filled with noise
        weights, 
        idelays,
        G,       # coupling scaling
        I,       # model params 
        Delta, 
        Gamma,
        eta,
        tau,
        J,       # end of model params
):

    def dr(r, V):
        dr = 1/tau * ( Delta / (np.pi * tau) + 2 * V * r)
        return dr
        
    def dV(r, V, r_c):
        dV = 1/tau * ( V**2 - np.pi**2 * tau**2 * r**2 + eta + J * tau * r + I + r_c ) 
        return dV

    def r_bound(r):
        return r if r >= 0. else 0. # max(0., r) is faster?

    for i in range(i0, i0 + nstep):
        for n in range(N):
            
            # coupling
            r_c = 0
            for m in range(N):
                r_c += weights[n,m] * r[m, i - idelays[n, m] - 1]
            r_c = r_c * G # post

            # precomputed additive noise 
            r_noise = r[n,i]
            V_noise = V[n,i]

            # Heun integration step
            dr_0 = dr(r[n,i-1], V[n,i-1]) 
            dV_0 = dV(r[n,i-1], V[n,i-1], r_c) 

            r_int = r[n,i-1] + dt*dr_0 + r_noise
            V_int = V[n,i-1] + dt*dV_0 + V_noise
            r_int = r_bound(r_int)

            r[n,i] = r[n,i-1] + dt*(dr_0 + dr(r_int, V_int))/2.0 + r_noise
            V[n,i] = V[n,i-1] + dt*(dV_0 + dV(r_int, V_int, r_c))/2.0 + V_noise
            r[n,i] = r_bound(r[n,i])

    return r, V
        





if __name__ == "__main__":
    from tvb.simulator.lab import *
    import matplotlib.pylab as plt
    import time

    G=0.1 # G=0.720 is too much -- integration fails numerically (NaN)
    conn_speed = 2.0
    dt = 0.01
    seed = 42
    nsigma = 0.01
    eta = -4.6
    J = 14.5
    Delta = 0.7
    tau = 1

    nstep = 100000 # 1000 ms

    conn = connectivity.Connectivity.from_file() # default 76 regions
    conn.speed = np.array([conn_speed])
    np.fill_diagonal(conn.weights, 0.)
    conn.weights = conn.weights/np.max(conn.weights)
    conn.configure()

    sim = simulator.Simulator(
        model=models.MontbrioPazoRoxin(
            eta   = np.r_[eta],
            J     = np.r_[J],
            Delta = np.r_[Delta],
            tau = np.r_[tau],
        ),
        connectivity=conn,
        coupling=coupling.Scaling(
          a=np.r_[G]
        ),
        conduction_speed=conn_speed,
        integrator=integrators.HeunStochastic(
          dt=dt,
          noise=noise.Additive(
              nsig=np.r_[nsigma, nsigma*2],
              noise_seed=seed
          )
        ),
        monitors=[
          monitors.Raw()
        ]
    )

    sim.configure()

    start = time.time()
    r, V = run_sim(sim, nstep)
    end = time.time()
    print( f'numba: {end - start:.2f} s')


    start = time.time()
    sim.run(simulation_length=nstep*dt)
    end = time.time()
    print( f'TVB: {end - start:.2f} s')

    plt.plot(r[:,::20].T)
    plt.show()
