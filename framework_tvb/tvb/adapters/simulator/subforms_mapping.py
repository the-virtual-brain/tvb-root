# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
from enum import Enum
from functools import partial

from tvb.core.entities.file.simulator.view_model import HeunDeterministicViewModel, HeunStochasticViewModel, \
    EulerDeterministicViewModel, EulerStochasticViewModel, RungeKutta4thOrderDeterministicViewModel, IdentityViewModel, \
    VODEViewModel, VODEStochasticViewModel, Dopri5ViewModel, Dopri5StochasticViewModel, Dop853ViewModel, \
    Dop853StochasticViewModel, AdditiveNoiseViewModel, MultiplicativeNoiseViewModel
from tvb.datatypes.equations import *

LINEAR_EQUATION = 'Linear'
GAUSSIAN_EQUATION = 'Gaussian'
DOUBLE_GAUSSIAN_EQUATION = 'Mexican-hat'
SIGMOID_EQUATION = 'Sigmoid'
GENRALIZED_SIGMOID_EQUATION = 'GeneralizedSigmoid'
SINUSOID_EQUATION = 'Sinusoid'
COSINE_EQUATION = 'Cosine'
ALPHA_EQUATION = 'Alpha'
PULSE_TRAIN_EQUATION = 'PulseTrain'


def get_ui_name_to_equation_dict():
    eq_name_to_class = {
        LINEAR_EQUATION: Linear,
        GAUSSIAN_EQUATION: Gaussian,
        DOUBLE_GAUSSIAN_EQUATION: DoubleGaussian,
        SIGMOID_EQUATION: Sigmoid,
        GENRALIZED_SIGMOID_EQUATION: GeneralizedSigmoid,
        SINUSOID_EQUATION: Sinusoid,
        COSINE_EQUATION: Cosine,
        ALPHA_EQUATION: Alpha,
        PULSE_TRAIN_EQUATION: PulseTrain
    }
    return eq_name_to_class


def get_ui_name_to_noise_dict():
    ui_name_to_noise = {
        'Additive': AdditiveNoiseViewModel,
        'Multiplicative': MultiplicativeNoiseViewModel
    }
    return ui_name_to_noise


def get_ui_name_to_integrator_dict():
    ui_name_to_integrator = {
        'Heun': HeunDeterministicViewModel,
        'Stochastic Heun': HeunStochasticViewModel,
        'Euler': EulerDeterministicViewModel,
        'Euler-Maruyama': EulerStochasticViewModel,
        'Runge-Kutta 4th order': RungeKutta4thOrderDeterministicViewModel,
        'Difference equation': IdentityViewModel,
        'Variable-order Adams / BDF': VODEViewModel,
        'Stochastic variable-order Adams / BDF': VODEStochasticViewModel,
        'Dormand-Prince, order (4, 5)': Dopri5ViewModel,
        'Stochastic Dormand-Prince, order (4, 5)': Dopri5StochasticViewModel,
        'Dormand-Prince, order 8 (5, 3)': Dop853ViewModel,
        'Stochastic Dormand-Prince, order 8 (5, 3)': Dop853StochasticViewModel,

    }
    return ui_name_to_integrator


class SubformsEnum(Enum):
    """
    Mapping between the type of subform and the method that should be called in order to get the corresponding object
    by its UI name.
    A Form that can be used as subform (eg: after a select field), should override get_subform_key() to return the
    proper key from this enum.
    """
    INTEGRATOR = partial(get_ui_name_to_integrator_dict)
    EQUATION = partial(get_ui_name_to_equation_dict)
    NOISE = partial(get_ui_name_to_noise_dict)
    # COUPLING = partial(get_ui_name_to_coupling_dict)
    # MONITOR = partial(get_ui_name_to_monitor_dict)
    # MODEL = partial(get_ui_name_to_model)
