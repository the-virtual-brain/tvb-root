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

The Data component of the Coupling datatypes.

.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""

import tvb.basic.traits.types_basic as basic
import tvb.datatypes.equations as equations

#TODO: This needs refactoring so that all specific types of Coupling are 
#      instances of a base Coupling class... 
class CouplingData(equations.Equation):
    """
    A base class for the Coupling datatypes
    """
    _ui_name = "Coupling"
    __generate_table__ = True


class LinearCouplingData(equations.Linear):
    """
    The equation for representing a Linear Coupling
    """
    _ui_name = "Linear Coupling"
    
    parameters = basic.Dict( 
        label = "Linear Coupling Parameters a and b",
        doc = """a: rescales connection strengths and maintains ratio and
                 b: shifts the base strength (maintain absolute difference)""",
        default = {"a": 0.00390625, "b": 0.0})
    
    __generate_table__ = True
        
        
class SigmoidalCouplingData(equations.GeneralizedSigmoid):
    """
    The equation for representing a Sigmoidal Coupling.
    """
    _ui_name = "Sigmoidal Coupling"
    __generate_table__ = True
    
    