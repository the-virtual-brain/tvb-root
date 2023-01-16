# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
A plain NumPy backend which uses templating to generate simulation
code.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os
import sys
import importlib
import numpy as np
import autopep8
import tempfile

from .templates import MakoUtilMix

from tvb.simulator.lab import *


class NpBackend(MakoUtilMix):

    def __init__(self):
        self.cgdir = tempfile.TemporaryDirectory()
        sys.path.append(self.cgdir.name)

    def build_py_func(self, template_source, content, name='kernel', print_source=False,
            modname=None, fname=None):
        "Build and retrieve one or more Python functions from template."
        source = self.render_template(template_source, content)
        source = autopep8.fix_code(source)
        if print_source:
            print(self.insert_line_numbers(source))
        if fname is not None:
            fullfname = os.path.join(self.cgdir.name, fname)
            with open(fullfname, 'w') as fd:
                fd.write(source)
        if modname is not None:
            return self.eval_module(source, name, modname)
        else:
            return self.eval_source(source, name, print_source)

    def eval_source(self, source, name, print_source):
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

    def _check_choices( self, val, choices):
        if not isinstance(val, choices):
            raise NotImplementedError("Unsupported simulator component. Given: {}\nExpected one of: {}".format(val, choices))

    def check_compatibility(self,sim):
        # monitors
        if len(sim.monitors) > 1:
            raise NotImplementedError("Configure with one monitor.")
        self._check_choices(sim.monitors[0], monitors.Raw)
        # integrators
        self._check_choices(sim.integrator, 
                (
                    integrators.HeunStochastic,
                    integrators.HeunDeterministic,
                    integrators.EulerStochastic,
                    integrators.EulerDeterministic,
                    integrators.Identity,
                    integrators.IdentityStochastic,
                    integrators.RungeKutta4thOrderDeterministic,
                )
        )
        # models 
        if sim.model.number_of_modes > 1:
            # this is a limitation e.g. by how nsig is now handled
            raise NotImplementedError("Only models with 1 mode are supported")
        self._check_choices(sim.model, models.MontbrioPazoRoxin) 
        # coupling
        self._check_choices(sim.coupling, 
                (coupling.Linear, coupling.Sigmoidal))
        # surface
        if sim.surface is not None:
            raise NotImplementedError("Surface simulation not supported.")
        # stimulus evaluated outside the backend, no restrictions

    def run_sim(self, sim, nstep=None, simulation_length=None, print_source=False):
        assert nstep is not None or simulation_length is not None or sim.simulation_length is not None

        self.check_compatibility(sim)

        if nstep is None:
            if simulation_length is None:
                simulation_length = sim.simulation_length
            nstep = int(np.ceil(simulation_length/sim.integrator.dt))

        buf = sim.history.buffer[...,0]
        rbuf = np.concatenate((buf[0:1], buf[1:][::-1]), axis=0)
        state = np.transpose(rbuf, (1, 0, 2)).astype('f')
        t = np.arange(1, nstep+1 ) * sim.integrator.dt 

        template = '<%include file="np-sim.py.mako"/>'
        content = dict(sim=sim, np=np, nstep=nstep)
        kernel = self.build_py_func(template, content, print_source=print_source)
        dX = state.copy()
        n_svar, _, n_node = state.shape
        state = state.reshape((n_svar, sim.connectivity.horizon, n_node))

        weights = sim.connectivity.weights.copy()
        yh = np.empty((len(t),)+state[:,0].shape)

        parmat = sim.model.spatial_parameter_matrix
        args = state, weights, yh, parmat
        if isinstance(sim.integrator, integrators.IntegratorStochastic):
            np.random.seed(sim.integrator.noise.noise_seed) 
            if len(sim.integrator.noise.nsig.shape) > 1:
                nsig = sim.integrator.noise.nsig[:,0] # no modes for now
            else:
                nsig = sim.integrator.noise.nsig
            args = args + (nsig,)
        if sim.connectivity.has_delays:
            args = args + (sim.connectivity.delay_indices,)
        kernel(*args)

        return (t, yh), 
