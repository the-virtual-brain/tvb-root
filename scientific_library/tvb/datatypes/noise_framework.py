# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
Module to handle framework specific methods related to noise sources.


.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import numpy.random

import tvb.basic.traits.parameters_factory as parameters_factory
from tvb.datatypes import equations
from tvb.simulator.noise import Noise

KEY_NOISE = "noise"
PARAMS_NOISE = (KEY_NOISE + '_parameters_')[:-1]
KEY_RANDOM_STREAM = "random_stream"
PARAMS_RANDOM_STREAM = (KEY_RANDOM_STREAM + '_parameters_')[:-1]
KEY_EQUATION = "b"
PARAMS_EQUATION = "b_parameters"


def build_noise(parent_parameters):
    """
    Build Noise entity from dictionary of parameters.
    :param parent_parameters: dictionary of parameters for the entity having Noise as attribute. \
                              The dictionary is after UI form-submit and framework pre-process.
    :return: Noise entity.

    """

    if KEY_NOISE not in parent_parameters:
        return None

    available_noise = parameters_factory.get_traited_subclasses(Noise)
    if 'Noise' not in available_noise:
        available_noise['Noise'] = Noise

    selected_noise = parent_parameters[KEY_NOISE]
    noise_params = parent_parameters[PARAMS_NOISE]
    del parent_parameters[KEY_NOISE]
    del parent_parameters[PARAMS_NOISE]

    if PARAMS_RANDOM_STREAM in noise_params:
        stream_params = noise_params[PARAMS_RANDOM_STREAM]
        random_stream = numpy.random.RandomState(seed=stream_params['init_seed'])
        del noise_params[PARAMS_RANDOM_STREAM]
        del noise_params[KEY_RANDOM_STREAM]
        noise_params[KEY_RANDOM_STREAM] = random_stream

    if PARAMS_EQUATION in noise_params:
        available_equations = parameters_factory.get_traited_subclasses(equations.Equation)
        eq_parameters = noise_params[PARAMS_EQUATION]["parameters"]
        equation = noise_params[KEY_EQUATION]
        equation = available_equations[equation](parameters=eq_parameters)
        del noise_params[PARAMS_EQUATION]
        del noise_params[KEY_EQUATION]
        noise_params[KEY_EQUATION] = equation

    noise_entity = available_noise[str(selected_noise)](**noise_params)
    parent_parameters[KEY_NOISE] = noise_entity
    return noise_entity


