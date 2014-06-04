import time
import itertools
from numpy import *
import numpy
from tvb.simulator import lab
# can't reload individual modules, must reboot ipython
from tvb.simulator.backend import driver_conf
driver_conf.using_gpu = 1
from tvb.simulator.backend import driver, util
reload(driver)

def makesim(speed=4.0):
    sim = lab.simulator.Simulator(
        model = lab.models.Generic2dOscillator(b=-10.0, c=0., d=0.02, I=0.0),
        connectivity = lab.connectivity.Connectivity(speed=speed),
        coupling = lab.coupling.Linear(a=1e-2),                                         # shape must match model..
        integrator = lab.integrators.EulerDeterministic(dt=2**-5),
        #integrator = lab.integrators.HeunStochastic(dt=2**-5, noise=lab.noise.Additive(nsig=ones((2, 1, 1))*1e-2)),
        monitors = lab.monitors.Raw()
    )
    sim.configure()
    return sim

def job(nsim=1, speed=4.0):
    print 'begin', nsim, speed
    sims = []
    for i, coupling_a in enumerate(r_[:0.1:1j*nsim]):
        for j, model_a in enumerate(r_[-2.0:2.0:32j]):
            simi = makesim(speed=speed)
            simi.coupling.a[:] = coupling_a
            simi.model.a[:] = model_a
            sims.append(simi)
    print 'simulations formulated'

    dh = driver.device_handler.init_like(sims[0])
    dh.n_thr = dh.n_rthr = len(sims)
    for i, simi in enumerate(sims):
        dh.fill_with(i, simi)
    print 'device is aligned'

    nsteps = 100
    ds = 50
    """
    ys1 = zeros((nsteps/ds, dh.n_node, dh.n_svar, len(sims)))
    ys2 = zeros((nsteps/ds, dh.n_node, dh.n_svar, len(sims)))
    dys1 = zeros((nsteps/ds, dh.n_node, dh.n_svar, len(sims)))
    dys2 = zeros((nsteps/ds, dh.n_node, dh.n_svar, len(sims)))
    """

    # just bench w/ one simulator
    simgens = [s(simulation_length=1000) for s in sims[0:1]]
    print 'simulation generators online'

    tc, tg = util.timer(), util.timer()
    _ = [next(sg) for sg in simgens]
    for i in range(nsteps):
        # iterate each simulation
        with tc:
            for j, (sgj, smj) in enumerate(zip(simgens, sims[0:1])):
                ((t, y), ) = next(sgj)
                # y.shape==(svar, nnode, nmode)
                """
                ys1[i/ds, ..., j] = y[..., 0].T
                dys1[i/ds, ..., j] = smj.dx[..., 0].T
                """
        with tg:
            dh()
        if i/ds and not i%ds:
            """
            ys2[i/ds, ...] = dh.x.device.get()
            dys2[i/ds, ...] = dh.dx1.device.get()
            err = ((ys1[:i/ds] - ys2[:i/ds])**2).sum()/ys1[:i/ds].ptp()/len(sims)
            """
            print t, tc.elapsed, tg.elapsed#, err

    return len(sims)*tc.elapsed/tg.elapsed

if __name__ == '__main__':
    import sys
    if len(sys.argv)<3:
        print 'benchmark_device.py nsim speed'
        sys.exit(0)

    print job(int(sys.argv[1]), float(sys.argv[2]))
