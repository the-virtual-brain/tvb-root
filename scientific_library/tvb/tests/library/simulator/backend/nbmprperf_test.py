import time
import numpy as np

from tvb.simulator import simulator, models, integrators, monitors, noise
from tvb.datatypes import connectivity
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from tvb.simulator.backend.nb import NbBackend


def make_sim(sim_len=1000.0):
    sim = simulator.Simulator(
        connectivity=connectivity.Connectivity.from_file(),
        model=models.MontbrioPazoRoxin(),
        integrator=integrators.HeunStochastic(
            dt=0.1,
            noise=noise.Additive(nsig=np.r_[0.001])),
        monitors=[monitors.Raw()],
        simulation_length=sim_len)
    sim.configure()
    return sim


def test_tvb_10ms(benchmark):
    sim = make_sim(10.0)
    benchmark(lambda : sim.run())


def test_tvb_100ms(benchmark):
    sim = make_sim(100.0)
    benchmark(lambda : sim.run())


def test_nb_pdq_10ms(benchmark):
    sim = make_sim(10.0)
    run_sim = NbMPRBackend().build_py_func(
        '<%include file="nb-montbrio.py.mako"/>', {}, name='run_sim')
    nstep = int(sim.simulation_length / sim.integrator.dt)
    benchmark(lambda : run_sim(sim, nstep))


def test_nb_pdq_100ms(benchmark):
    sim = make_sim(100.0)
    run_sim = NbMPRBackend().build_py_func(
        '<%include file="nb-montbrio.py.mako"/>', {}, name='run_sim')
    nstep = int(sim.simulation_length / sim.integrator.dt)
    benchmark(lambda : run_sim(sim, nstep))


def test_nb_mako_10ms(benchmark):
    sim = make_sim(10.0)
    template = '<%include file="nb-sim.py.mako"/>'
    content = dict(sim=sim, np=np, debug_nojit=False)
    kernel = NbBackend().build_py_func(template, content, print_source=True, name='run_sim')
    benchmark(lambda : kernel(sim))


def test_nb_mako_100ms(benchmark):
    sim = make_sim(100.0)
    template = '<%include file="nb-sim.py.mako"/>'
    content = dict(sim=sim, np=np, debug_nojit=False)
    kernel = NbBackend().build_py_func(template, content, print_source=True, name='run_sim')
    benchmark(lambda : kernel(sim))
