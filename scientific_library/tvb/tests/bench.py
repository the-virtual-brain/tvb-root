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

"""
This file benchmarks isolated components in the scientific library.

"""

import sys
import time
import importlib
import numpy
import cpuinfo

print('CPU is a %s' % (cpuinfo.get_cpu_info()['brand'],))


def get_subclasses(submodule_name, base_class_name, ignore_names=None):
    modname = 'tvb.simulator.' + submodule_name
    mod = importlib.import_module(modname, modname)
    base_class = getattr(mod, base_class_name)
    ignore_names = [] if ignore_names is None else ignore_names
    for key in dir(mod):
        attr = getattr(mod, key)
        if (key != base_class_name
                and isinstance(attr, type)
                and issubclass(attr, base_class)
                and key not in ignore_names):
            yield attr


# integrators {{{

def integrators():
    return get_subclasses('integrators', 'Integrator', ignore_names=["IntegratorStochastic"])


def nop_dfun(x, coupling, local_coupling):
    return x


def eps_for_integrator(integrator_class, n_node, time_limit=0.5):
    integ = integrator_class()
    integ.configure()
    if 'Stochastic' in integrator_class.__name__:
        integ.noise.dt = integ.dt
    x = numpy.random.randn(n_node)
    thunk = lambda: integ.scheme(x, nop_dfun, None, None, 0.0)
    thunk()
    # start timing
    tic = time.time()
    n_eval = 0
    while (time.time() - tic) < time_limit:
        thunk()
        n_eval += 1
    toc = time.time()
    return n_eval / (toc - tic)


# }}}

# models {{{ 

def models():
    return get_subclasses('models', 'Model')


def randn_state_for_model(model, n_node):
    shape = (model.nvar, n_node, model.number_of_modes)
    state = numpy.random.randn(*shape)
    return state


def zero_coupling_for_model(model, n_node):
    n_cvar = len(model.cvar)
    shape = (n_cvar, n_node, model.number_of_modes)
    coupling = numpy.zeros(shape)
    return coupling


def eps_for_model(model_class, n_node, time_limit=0.5):
    model = model_class()
    model.configure()
    state = randn_state_for_model(model, n_node)
    coupling = zero_coupling_for_model(model, n_node)
    # throw one away in case of initialization
    model.dfun(state, coupling)
    # start timing
    tic = time.time()
    n_eval = 0
    while (time.time() - tic) < time_limit:
        model.dfun(state, coupling)
        n_eval += 1
    toc = time.time()
    return n_eval / (toc - tic)


# }}}

def eps_report_for_components(comps, eps_func):
    n_nodes = [2 << i for i in range(14)]
    sys.stdout.write('%30s' % ('n_node',))
    [sys.stdout.write('%06s' % (n,)) for n in n_nodes]
    sys.stdout.write('\n')
    sys.stdout.flush()
    for comp in comps:
        name = comp.__name__ if isinstance(comp, type) else comp.__class__.__name__
        sys.stdout.write('%30s' % (name,))
        for n_node in n_nodes:
            deps = eps_func(comp, n_node)
            sdeps = '%0.1f' % (deps / 1e3,)
            sys.stdout.write('%06s' % (sdeps,))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    print('units in kHz')
    print('benchmarking models')
    eps_report_for_components(models(), eps_for_model)
    print('benchmarking integrators')
    from tvb.simulator.integrators import RungeKutta4thOrderDeterministic

    integs = list(integrators()) + [RungeKutta4thOrderDeterministic]
    eps_report_for_components(integs, eps_for_integrator)

# vim: sw=4 sts=4 ai et foldmethod=marker
