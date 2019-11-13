# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
A collection of neuronal dynamics models.

Specific models inherit from the abstract class Model.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Gaurav Malhotra <Gaurav@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

# we don't import all models by default here, since they are time consuming
# to setup (e.g. numba gufunc compilation), but provide them as module-level
# properties for compatibility with previous version of TVB. For example
# 
#     import tvb.simulator.models.Epileptor
# 
# works, but only lazily loads the tvb.simulator.models.epileptor module
# and returns the Epileptor class.

_module_models = {
    'base': 'Model'.split(', '),
    'epileptor': 'Epileptor, Epileptor2D'.split(', '),
    'epileptor_rs': 'EpileptorRestingState'.split(', '),
    'epileptorcodim3': 'EpileptorCodim3, EpileptorCodim3SlowMod'.split(', '),
    'hopfield': 'Hopfield'.split(', '),
    'jansen_rit': 'JansenRit, ZetterbergJansen'.split(', '),
    'larter_breakspear': 'LarterBreakspear'.split(', '),
    'linear': 'Linear'.split(', '),
    'oscillator': 'Generic2dOscillator, Kuramoto, SupHopf'.split(', '),
    'stefanescu_jirsa': 'ReducedSetFitzHughNagumo, ReducedSetHindmarshRose'.split(', '),
    'wilson_cowan': 'WilsonCowan'.split(', '),
    'wong_wang': 'ReducedWongWang'.split(', '),
    'wong_wang_exc_inh': 'ReducedWongWangExcInh'.split(', '),
    'zerlaut': 'ZerlautFirstOrder, ZerlautSecondOrder'.split(', '),
}


def _delay_import_one(mod, model):
    """Create getter thunk for module and model name.
    """
    import importlib
    def do_import(_):
        module_name = f'tvb.simulator.models.{mod}'
        model_module = importlib.import_module(module_name)
        return getattr(model_module, model)
    return property(do_import)


def _delay_model_imports():
    """Set up this module with all properties for all models.
    """
    # create substitute module class & object
    class _Module:
        pass
    module = _Module()
    module.__dict__ = globals()
    # create properties for each model
    props = {}
    for mod, models in _module_models.items():
        for model in models:
            setattr(_Module, model, _delay_import_one(mod, model))
    # register module object
    import sys
    module._module = sys.modules[module.__name__]
    module._pmodule = module
    sys.modules[module.__name__] = module


_delay_model_imports()
