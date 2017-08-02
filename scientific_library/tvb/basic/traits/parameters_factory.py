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
#
#
"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import numpy
import six
from tvb.basic.traits.core import TYPE_REGISTER

KEYWORD_PARAMS = "_parameters_"
KEYWORD_SEPARATOR = "_"
KEYWORD_OPTION = "option_"


def get_traited_subclasses(parent_class):
    """
    :param parent_class: SuperClass, to return valid sub-classes of this (e.g. Model).
    :return: {class_name: sub_class_instance}
        e.g. {'WilsonCowan': WilsonCowan, ....}
    """
    classes_list = TYPE_REGISTER.subclasses(parent_class)
    result = {}
    for class_instance in classes_list:
        result[class_instance.__name__] = class_instance
    return result


def get_traited_instance_for_name(class_name, parent_class, params_dictionary):
    """
    :param class_name: Short Traited Class name.
    :param parent_class: Traited basic type expected (e.g. Model)
    :param params_dictionary: dictionary of parameters to be passed to the constructor of the class.
    :return: Class instantiated corresponding to the given name (e.g. FHN() )
    """
    available_subclasses = get_traited_subclasses(parent_class)
    if class_name not in available_subclasses:
        return None
    class_instance = available_subclasses[class_name]
    entity = class_instance(**params_dictionary)
    return entity


def collapse_params(args, simple_select_list, parent=''):
    """ In case of variables with similar names:
    (name_parameters_[option_xx]_paramKey) collapse then into dictionary
    of parameters. This is used after parameters POST, on Operation Launch.
    """
    result = {}
    for name, value in args.items():
        short_name = name
        option = None
        key = None
        if name.find(KEYWORD_PARAMS) >= 0:
            short_name = name[0: (name.find(KEYWORD_PARAMS) + 11)]
            key = name[(name.find(KEYWORD_PARAMS) + 12):]
            if key.find(KEYWORD_OPTION) == 0:
                key = key[7:]  # Remove '_option_'
                option = key[0: key.find(KEYWORD_SEPARATOR)]
                key = key[key.find(KEYWORD_SEPARATOR) + 1:]

        if key is None:
            result[name] = value
        else:
            if short_name not in result:
                result[short_name] = {}
            if option is None:
                result[short_name][key] = value
            else:
                if option not in result[short_name]:
                    result[short_name][option] = {}
                result[short_name][option][key] = value

    for level1_name, level1_params in result.items():
        if KEYWORD_PARAMS[:-1] in level1_name and isinstance(level1_params, dict):
            short_parent_name = level1_name[0: level1_name.find(KEYWORD_PARAMS) - 10]
            if (parent + short_parent_name) in simple_select_list:
                # simple select
                if isinstance(result[short_parent_name], (str, unicode)):
                    parent_prefix = level1_name + KEYWORD_SEPARATOR + KEYWORD_OPTION
                    parent_prefix += result[short_parent_name]
                    parent_prefix += KEYWORD_SEPARATOR
                    # Ignore options in case of simple selects
                    # Take only attributes for current selected option.
                    if result[short_parent_name] in level1_params:
                        level1_params = level1_params[result[short_parent_name]]
                    else:
                        level1_params = {}
                else:
                    parent_prefix = level1_name

                transformed_params = collapse_params(level1_params, simple_select_list, parent + parent_prefix)
                result[level1_name] = transformed_params
            elif short_parent_name in result:
                # multiple select
                for level2_name, level2_params in level1_params.items():
                    parent_prefix = level1_name + KEYWORD_SEPARATOR + KEYWORD_OPTION
                    parent_prefix += level2_name + KEYWORD_SEPARATOR
                    transformed_params = collapse_params(level2_params, simple_select_list, parent + parent_prefix)
                    result[level1_name][level2_name] = transformed_params
    return result


def try_parse(val):
    if isinstance(val, dict):
        return {str(k): try_parse(v) for k, v in six.iteritems(val)}
    if isinstance(val, list):
        return val

    try:
        return int(val)
    except Exception:
        try:
            return float(val)
        except Exception:
            if val.find('[') == 0:
                try:
                    return numpy.array(val.replace('[', '').replace(']', '').split(','), dtype=numpy.float64)
                except Exception:
                    pass
            return str(val)
