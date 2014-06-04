
import os
import unittest
import itertools

import numpy

try:
    import pycuda
    _pycuda_import_error = None
except Exception as _pycuda_import_error:
    pycuda = None


class BaseCUDATest(unittest.TestCase):

    def setUp(self):
        from tvb.simulator import lab
        from tvb.simulator.backend import driver_conf
        driver_conf.using_gpu = 1
        from tvb.simulator.backend import driver, util
        reload(driver)
        for k in 'lab driver util'.split():
            setattr(self, k, locals()[k])


class TestRegionParallel(BaseCUDATest):
    "Tests CUDA backend for simple parameter sweep in region based simulation"

    # parameters to sweep (coupling & model $a$)
    cas = numpy.r_[:0.1:8j]
    mas = numpy.r_[-2.0:2.0:8j]

    # sim & output parameters
    nsteps = 10000
    ds = 50

    def setUp(self):
        super(TestRegionParallel, self).setUp()
        self.build_all_sims()
        self.pack_device()
        self.setup_outputs()
        self.init_sims()

    def new_proto_sim(self, ca, ma):
        lab = self.lab
        sim = lab.simulator.Simulator(
            model = lab.models.Generic2dOscillator(b=-10.0, c=0., d=0.02, I=0.0),
            connectivity = lab.connectivity.Connectivity(speed=4.0),
            coupling = lab.coupling.Linear(a=1e-2),                                         # shape must match model..
            integrator = lab.integrators.HeunDeterministic(dt=2**-5),
            #integrator = lab.integrators.HeunStochastic(dt=2**-5, noise=lab.noise.Additive(nsig=ones((2, 1, 1))*1e-2)),
            monitors = lab.monitors.Raw()
        )
        sim.configure()
        sim.model.a[:] = ma
        sim.coupling.a[:] = ca
        return sim

    def build_all_sims(self):
        self.sims = [self.new_proto_sim(ca, ma) for ca, ma in itertools.product(self.cas, self.mas)]

    def pack_device(self):
        self.dh = self.driver.device_handler.init_like(self.sims[0])
        self.dh.n_thr = self.dh.n_rthr = len(self.sims)
        for i, simi in enumerate(self.sims):
            self.dh.fill_with(i, simi)

    def setup_outputs(self):
        sh = self.nsteps/self.ds, self.dh.n_node, self.dh.n_svar, len(self.sims)
        self.ys1, self.ys2 = numpy.zeros((2,) + sh)

    def init_sims(self):
        self.simgens = [s(simulation_length=1000) for s in self.sims]

    def integrate(self):
        ds = self.ds
        for i in range(self.nsteps):
            # step all simulations
            for j, (sgj, smj) in enumerate(zip(self.simgens, self.sims)):
                ((t, y), ) = next(sgj)
                self.ys1[i/ds, ..., j] = y[..., 0].T
            self.dh()
            self.ys2[i/ds, ...] = self.dh.x.device.get()
            if i and i % ds == 0:
                err = ((self.ys1[:i/ds] - self.ys2[:i/ds])**2).sum()/self.ys1[:i/ds].ptp()/len(self.sims)
                self.assertLess(err, 40.0)

    def check_parsweep_error(self):
        here = os.path.dirname(os.path.abspath(__file__))
        err_par_accept = numpy.load(os.path.join(here, 'cuda-err-lim.npy'))
        err = (self.ys1 - self.ys2)**2
        err_par_meas = err.reshape((-1, err.shape[-1])).sum(axis=0)
        for meas, accept in zip(err_par_meas, err_par_accept):
            self.assertLessEqual(meas, 2*accept)

    @unittest.skipIf(pycuda is None,
                     "importing pycuda failed with %r" % (_pycuda_import_error,))
    def test_run(self):
        self.integrate()
        self.check_parsweep_error()

    # uncomment to debug interactively
    """
    def runTest(self):
        self.test_run()
    """

if __name__ == '__main__':
    unittest.main()
