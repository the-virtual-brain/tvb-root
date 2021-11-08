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

    def check_compatibility(self, sim): 
        def check_choices(val, choices):
            if not isinstance(val, choices):
                raise NotImplementedError("Unsupported simulator component. Given: {}\nExpected one of: {}".format(val, choices))
        # monitors
        if len(sim.monitors) > 1:
            raise NotImplementedError("Configure with one monitor.")
        check_choices(sim.monitors[0], (monitors.Raw, monitors.TemporalAverage))
        # integrators
        check_choices(sim.integrator, integrators.HeunStochastic)
        # models
        check_choices(sim.model, models.MontbrioPazoRoxin)
        # coupling
        check_choices(sim.coupling, coupling.Linear)
        # surface
        if sim.surface is not None:
            raise NotImplementedError("Surface simulation not supported.")
        # stimulus evaluated outside the backend, no restrictions


    def run_sim(self, sim, nstep=None, simulation_length=None, chunksize=100000, compatibility_mode=False):
        assert nstep is not None or simulation_length is not None or sim.simulation_length is not None

        self.check_compatibility(sim)
        if nstep is None:
            if simulation_length is None:
                simulation_length = sim.simulation_length
            nstep = int(np.ceil(simulation_length/sim.integrator.dt))

        if isinstance(sim.monitors[0], monitors.Raw):
            svar_bufs = self._run_sim_plain(sim, nstep, compatibility_mode=compatibility_mode)
            time = np.arange(svar_bufs[0].shape[1]) * sim.integrator.dt
        elif isinstance(sim.monitors[0], monitors.TemporalAverage):
            svar_bufs = self._run_sim_tavg_chunked(sim, nstep, chunksize=chunksize, compatibility_mode=compatibility_mode)
            T = sim.monitors[0].period
            time = np.arange(svar_bufs[0].shape[1]) * T + 0.5 * T
        else:
            raise NotImplementedError("Only Raw or TemporalAverage monitors supported.")
        data = np.concatenate(
                [svar_buf.T[:,np.newaxis,:, np.newaxis] for svar_buf in svar_bufs],
                axis=1
        )
        return (time, data),   

    def _run_sim_plain(self, sim, nstep=None, compatibility_mode=False):
        template = '<%include file="nb-montbrio.py.mako"/>'
        content = dict(
                compatibility_mode=compatibility_mode, 
                sim=sim
        ) 
        integrate = self.build_py_func(template, content, name='integrate', print_source=True)

        horizon = sim.connectivity.horizon
        buf_len = horizon + nstep
        N = sim.connectivity.number_of_regions
        gf = sim.integrator.noise.gfun(None)

        svar_bufs = [buf for buf in sim.integrator.noise.generate( shape=(sim.model.nvar,N,buf_len) ) * gf]
        for i, svar_buf in enumerate(svar_bufs):
            svar_buf[:,:horizon] = np.roll(sim.history.buffer[:,i,:,0], -1, axis=0).T

        if sim.stimulus is None:
            stimulus = None
        else:
            sim.stimulus.configure_space()
            sim.stimulus.configure_time(np.arange(nstep)*sim.integrator.dt)
            stimulus = sim.stimulus()

        svar_bufs = integrate(
            N = N,
            dt = sim.integrator.dt,
            nstep = nstep,
            i0 = horizon,
            **dict(zip(sim.model.state_variables,svar_bufs)),
            weights = sim.connectivity.weights, 
            idelays = sim.connectivity.idelays,
            parmat = sim.model.spatial_parameter_matrix.T,
            stimulus = stimulus
        )
        return [svar_buf[:,horizon:] for svar_buf in svar_bufs]

    def _time_average(self, ts, istep):
        N, T = ts.shape
        return np.mean(ts.reshape(N,T//istep,istep),-1) # length of ts better be multiple of istep 

    def _run_sim_tavg_chunked(self, sim, nstep, chunksize, compatibility_mode=False):
        template = '<%include file="nb-montbrio.py.mako"/>'
        content = dict(sim=sim, compatibility_mode=compatibility_mode) 
        integrate = self.build_py_func(template, content, name='integrate', print_source=False)
        # chunksize in number of steps 
        horizon = sim.connectivity.horizon
        N = sim.connectivity.number_of_regions
        gf = sim.integrator.noise.gfun(None)

        tavg_steps=sim.monitors[0].istep
        assert tavg_steps < chunksize
        assert chunksize % tavg_steps == 0
        tavg_chunksize = chunksize // tavg_steps


        assert nstep % tavg_steps == 0
        svar_outs = [svar_out for svar_out in np.zeros((sim.model.nvar,N,nstep//tavg_steps))]

        svar_bufs = [buf for buf in sim.integrator.noise.generate( shape=(sim.model.nvar,N,chunksize+horizon) ) * gf]
        for i, svar_buf in enumerate(svar_bufs):
            svar_buf[:,:horizon] = np.roll(sim.history.buffer[:,i,:,0], -1, axis=0).T


        for chunk, _ in enumerate(range(horizon, nstep+horizon, chunksize)):
            if sim.stimulus is None:
                stimulus = None
            else:
                sim.stimulus.configure_space()
                sim.stimulus.configure_time(
                        np.arange(chunk*chunksize, (chunk+1)*chunksize)*sim.integrator.dt
                )
                stimulus = sim.stimulus()
            svar_bufs = integrate(
                N = N,
                dt = sim.integrator.dt,
                nstep = chunksize,
                i0 = horizon,
                **dict(zip(sim.model.state_variables,svar_bufs)),
                weights = sim.connectivity.weights, 
                idelays = sim.connectivity.idelays,
                parmat = sim.model.spatial_parameter_matrix,
                stimulus = stimulus
            )

            tavg_chunk = chunk * tavg_chunksize
            svar_chunks = [self._time_average(svar[:, horizon:], tavg_steps) for svar in svar_bufs]
            if tavg_chunk+tavg_chunksize > svar_outs[0].shape[1]:
                cutoff = tavg_chunk+tavg_chunksize - svar_outs[0].shape[1]
                for svar_chunk in svar_chunks:
                    svar_chunk = svar_chunk[:,:-cutoff]
                    
            for svar_out, svar_chunk in zip (svar_outs, svar_chunks):
                svar_out[:,tavg_chunk:tavg_chunk+tavg_chunksize] = svar_chunk

            for svar_buf in svar_bufs:
                svar_buf[:,:horizon] = svar_buf[:,-horizon:]


            for svar_buf, svar_noise in zip(svar_bufs, sim.integrator.noise.generate( shape=(sim.model.nvar,N,chunksize) ) * gf):
                svar_buf[:,horizon:] = svar_noise


        return svar_outs
