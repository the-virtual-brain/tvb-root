# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

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

def test_nb_mako_1000ms(benchmark):
    sim = make_sim(1000.0)
    template = '<%include file="nb-sim.py.mako"/>'
    content = dict(sim=sim, np=np, debug_nojit=False)
    kernel = NbBackend().build_py_func(template, content, print_source=True, name='run_sim')
    benchmark(lambda : kernel(sim))

def run_sim_pdq(sim, integrate):
    nstep = int(np.ceil(sim.simulation_length/sim.integrator.dt))
    horizon = sim.connectivity.horizon
    buf_len = horizon + nstep
    N = sim.connectivity.number_of_regions
    gf = sim.integrator.noise.gfun(None)

    r, V = sim.integrator.noise.generate( shape=(2,N,buf_len) ) * gf
    r[:,:horizon] = np.roll(sim.history.buffer[:,0,:,0], -1, axis=0).T
    V[:,:horizon] = np.roll(sim.history.buffer[:,1,:,0], -1, axis=0).T


    r, V = integrate(
        N = N,
        dt = sim.integrator.dt,
        nstep = nstep,
        i0 = horizon,
        r=r,
        V=V,
        weights = sim.connectivity.weights, 
        idelays = sim.connectivity.idelays,
        parmat = sim.model.spatial_parameter_matrix,
        stimulus = None
    )
    return r[:,horizon:], V[:,horizon:]

def test_nb_pdq_10ms(benchmark):
    sim = make_sim(10.0)
    template = '<%include file="nb-montbrio.py.mako"/>'
    content = dict(
            compatibility_mode=False, 
            sim=sim
    ) 
    integrate = NbMPRBackend().build_py_func(template, content, name='integrate', print_source=True)

    benchmark(lambda : run_sim_pdq(sim, integrate))

def test_nb_pdq_100ms(benchmark):
    sim = make_sim(100.0)
    template = '<%include file="nb-montbrio.py.mako"/>'
    content = dict(
            compatibility_mode=False, 
            sim=sim
    ) 
    integrate = NbMPRBackend().build_py_func(template, content, name='integrate', print_source=True)

    benchmark(lambda : run_sim_pdq(sim, integrate))

def test_nb_pdq_1000ms(benchmark):
    sim = make_sim(1000.0)
    template = '<%include file="nb-montbrio.py.mako"/>'
    content = dict(
            compatibility_mode=False, 
            sim=sim
    ) 
    integrate = NbMPRBackend().build_py_func(template, content, name='integrate', print_source=True)

    benchmark(lambda : run_sim_pdq(sim, integrate))
