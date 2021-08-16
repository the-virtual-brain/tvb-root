# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
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

"""
Numba backend which uses templating to generate simulation
code.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os
import importlib
import numpy as np
import autopep8

from .templates import MakoUtilMix
from tvb.simulator.lab import *


class NbMPRBackend(MakoUtilMix):

    def build_py_func(self, template_source, content, name='kernel', print_source=False,
            modname=None):
        "Build and retrieve one or more Python functions from template."
        source = self.render_template(template_source, content)
        source = autopep8.fix_code(source)
        if print_source:
            print(self.insert_line_numbers(source))
        if modname is not None:
            return self.eval_module(source, name, modname)
        else:
            return self.eval_source(source, name)

    def eval_source(self, source, name):
        globals_ = {}
        try:
            exec(source, globals_)
        except Exception as exc:
            if not print_source:
                print(self._insert_line_numbers(source))
            raise exc
        fns = [globals_[n] for n in name.split(',')]
        return fns[0] if len(fns)==1 else fns

    def eval_module(self, source, name, modname):
        here = os.path.abspath(os.path.dirname(__file__))
        genp = os.path.join(here, 'templates', 'generated')
        with open(f'{genp}/{modname}.py', 'w') as fd:
            fd.write(source)
        fullmodname = f'tvb.simulator.backend.templates.generated.{modname}'
        mod = importlib.import_module(fullmodname)
        fns = [getattr(mod,n) for n in name.split(',')]
        return fns[0] if len(fns)==1 else fns

    def run_sim(self, sim, nstep=None, simulation_length=None, chunksize=100000, compatibility_mode=False):
        assert nstep is not None or simulation_length is not None or sim.simulation_length is not None
        if nstep is None:
            if simulation_length is None:
                simulation_length = sim.simulation_length
            nstep = int(np.ceil(simulation_length/sim.integrator.dt))

        assert len(sim.monitors) == 1, "Configure with exatly one monitor."
        if isinstance(sim.monitors[0], monitors.Raw):
            r, V = self._run_sim_plain(sim, nstep, compatibility_mode=compatibility_mode)
            time = np.arange(r.shape[1]) * sim.integrator.dt
        elif isinstance(sim.monitors[0], monitors.TemporalAverage):
            r, V = self._run_sim_tavg_chunked(sim, nstep, chunksize=chunksize, compatibility_mode=compatibility_mode)
            T = sim.monitors[0].period
            time = np.arange(r.shape[1]) * T + 0.5 * T
        else:
            raise NotImplementedError("Only Raw or TemporalAverage monitors supported.")
        data = np.concatenate(
                (r.T[:,np.newaxis,:, np.newaxis], V.T[:,np.newaxis,:, np.newaxis]),
                axis=1
        )
        return (time, data),   

    def _run_sim_plain(self, sim, nstep=None, compatibility_mode=False):
        template = '<%include file="nb-montbrio.py.mako"/>'
        content = dict(
                compatibility_mode=compatibility_mode, 
                sim=sim
        ) 
        integrate = self.build_py_func(template, content, name='_mpr_integrate', print_source=True)

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
            G = sim.coupling.a.item(),
            parmat = sim.model.spatial_parameter_matrix.T
        )
        return r[:,horizon:], V[:,horizon:]

    def _time_average(self, ts, istep):
        N, T = ts.shape
        return np.mean(ts.reshape(N,T//istep,istep),-1) # length of ts better be multiple of istep 

    def _run_sim_tavg_chunked(self, sim, nstep, chunksize, compatibility_mode=False):
        template = '<%include file="nb-montbrio.py.mako"/>'
        content = dict(sim=sim, compatibility_mode=compatibility_mode) 
        integrate = self.build_py_func(template, content, name='_mpr_integrate', print_source=False)
        # chunksize in number of steps 
        horizon = sim.connectivity.horizon
        N = sim.connectivity.number_of_regions
        gf = sim.integrator.noise.gfun(None)

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
            r, V = integrate(
                N = N,
                dt = sim.integrator.dt,
                nstep = chunksize,
                i0 = horizon,
                r=r,
                V=V,
                weights = sim.connectivity.weights, 
                idelays = sim.connectivity.idelays,
                G = sim.coupling.a.item(),
                parmat = sim.model.spatial_parameter_matrix
            )

            tavg_chunk = chunk * tavg_chunksize
            r_chunk = self._time_average(r[:, horizon:], tavg_steps)
            V_chunk = self._time_average(V[:, horizon:], tavg_steps)
            if tavg_chunk+tavg_chunksize > r_out.shape[1]:
                cutoff = tavg_chunk+tavg_chunksize - r_out.shape[1]
                r_chunk = r_chunk[:,:-cutoff]
                V_chunk = V_chunk[:,:-cutoff]
            
            r_out[:,tavg_chunk:tavg_chunk+tavg_chunksize] = r_chunk
            V_out[:,tavg_chunk:tavg_chunk+tavg_chunksize] = V_chunk

            r[:,:horizon] = r[:,-horizon:]
            V[:,:horizon] = V[:,-horizon:]

            r_noise, V_noise = sim.integrator.noise.generate( shape=(2,N,chunksize) ) * gf
            r[:,horizon:] = r_noise
            V[:,horizon:] = V_noise

        return r_out, V_out
