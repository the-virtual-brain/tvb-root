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

from enum import Enum


class ModelsEnum(Enum):
    BASE_MODEL = "Model"
    EPILEPTOR = "Epileptor"
    EPILEPTOR_2D = "Epileptor2D"
    EPILEPTOR_RS = "EpileptorRestingState"
    EPILEPTOR_CODIM_3 = "EpileptorCodim3"
    EPILEPTOR_CODIM_3_SLOW = "EpileptorCodim3SlowMod"
    HOPFIELD = "Hopfield"
    JANSEN_RIT = "JansenRit"
    ZETTERBERG_JANSEN = "ZetterbergJansen"
    LARTER_BREAKSPEAR = "LarterBreakspear"
    LINEAR = "Linear"
    GENERIC_2D_OSCILLATOR = "Generic2dOscillator"
    KURAMOTO = "Kuramoto"
    SUP_HOPF = "SupHopf"
    REDUCED_SET_FITZ_HUGH_NAGUMO = "ReducedSetFitzHughNagumo"
    REDUCED_SET_HINDMARSH_ROSE = "ReducedSetHindmarshRose"
    WILSON_COWAN = "WilsonCowan"
    REDUCED_WONG_WANG = "ReducedWongWang"
    REDUCED_WONG_WANG_EXC_INH = "ReducedWongWangExcInh"
    ZERLAUT_FIRST_ORDER = "ZerlautAdaptationFirstOrder"
    ZERLAUT_SECOND_ORDER = "ZerlautAdaptationSecondOrder"
    MONTBRIO_PAZO_ROXIN = "MontbrioPazoRoxin"
    COOMBES_BYRNE = "CoombesByrne"
    COOMBES_BYRNE_2D = "CoombesByrne2D"
    GAST_SCHMIDT_KNOSCHE_SD = "GastSchmidtKnosche_SD"
    GAST_SCHMIDT_KNOSCHE_SF = "GastSchmidtKnosche_SF"
    DUMONT_GUTKIN = "DumontGutkin"
    DECO_BALANCED_EXC_INH = "DecoBalancedExcInh"

    def get_class(self):
        return _get_imported_model(self.value)

    @staticmethod
    def get_base_model_subclasses():
        return [model.get_class() for model in list(ModelsEnum) if model != ModelsEnum.BASE_MODEL]


def _get_imported_model(model):
    import sys
    # Imported modules
    imported_modules = sys.modules['tvb.simulator.models']
    try:
        return getattr(imported_modules, model)
    except AttributeError:
        return None


_module_models = {
    'base': [ModelsEnum.BASE_MODEL],
    'epileptor': [ModelsEnum.EPILEPTOR, ModelsEnum.EPILEPTOR_2D],
    'epileptor_rs': [ModelsEnum.EPILEPTOR_RS],
    'epileptorcodim3': [ModelsEnum.EPILEPTOR_CODIM_3, ModelsEnum.EPILEPTOR_CODIM_3_SLOW],
    'hopfield': [ModelsEnum.HOPFIELD],
    'jansen_rit': [ModelsEnum.JANSEN_RIT, ModelsEnum.ZETTERBERG_JANSEN],
    'larter_breakspear': [ModelsEnum.LARTER_BREAKSPEAR],
    'linear': [ModelsEnum.LINEAR],
    'oscillator': [ModelsEnum.GENERIC_2D_OSCILLATOR, ModelsEnum.KURAMOTO, ModelsEnum.SUP_HOPF],
    'stefanescu_jirsa': [ModelsEnum.REDUCED_SET_HINDMARSH_ROSE, ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO],
    'wilson_cowan': [ModelsEnum.WILSON_COWAN],
    'wong_wang': [ModelsEnum.REDUCED_WONG_WANG],
    'wong_wang_exc_inh': [ModelsEnum.REDUCED_WONG_WANG_EXC_INH, ModelsEnum.DECO_BALANCED_EXC_INH],
    'zerlaut': [ModelsEnum.ZERLAUT_FIRST_ORDER, ModelsEnum.ZERLAUT_SECOND_ORDER],
    'infinite_theta': [ModelsEnum.MONTBRIO_PAZO_ROXIN, ModelsEnum.COOMBES_BYRNE, ModelsEnum.COOMBES_BYRNE_2D, ModelsEnum.GAST_SCHMIDT_KNOSCHE_SF, ModelsEnum.GAST_SCHMIDT_KNOSCHE_SD, ModelsEnum.DUMONT_GUTKIN],
}


def _find_lems_models():
    import os, tvb.dsl
    dsl_path = os.path.dirname(os.path.abspath(tvb.dsl.__file__))
    xml_folder = os.path.join(dsl_path, 'NeuroML', 'XMLmodels')
    for xml_filename in os.listdir(xml_folder):
        # try to get model name
        fullfilename = os.path.join(xml_folder, xml_filename)
        print(fullfilename)
        with open(fullfilename, 'r') as fd:
            for line in fd.readlines():
                if '<ComponentType name=' in line:
                    _, name_, *_ = line.strip().split()
                    _, name, _ = name_.split("=")[1].split('"')
                    break


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
    for mod, models in _module_models.items():
        for model in models:
            setattr(_Module, model.value, _delay_import_one(mod, model.value))
    # register module object
    import sys
    module._module = sys.modules[module.__name__]
    module._pmodule = module
    sys.modules[module.__name__] = module


_delay_model_imports()
