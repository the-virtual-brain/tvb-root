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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.basic.traits.core import TYPE_REGISTER



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


    