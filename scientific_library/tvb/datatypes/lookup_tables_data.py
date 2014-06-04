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

The Data component of the LookUpTables datatypes. These are intended to be 
tables containing the values of computationally expensive functions used
within the tvb.simulator.models.

At present, we only make use of these in the Brunel and Wang model.

.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""

import numpy
import tvb.datatypes.arrays as arrays
import tvb.basic.traits.types_basic as basic
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.types_mapped import MappedType


LOG = get_logger(__name__)

# NOTE: For the time being, we make use of the precalculated tables we already have.
# however, we LookUpTables datatypes could have a compute() method to 
# calculate a given function (using Equation datatypes? ). 


class LookUpTableData(MappedType):
    """
    Lookup Tables for storing pre-computed functions.
    Specific table subclasses are implemented below.
    """

    _base_classes = ['LookUpTables']

    equation = basic.String(
        label="String representation of the precalculated function",
        doc="""A latex representation of the function whose values are stored
            in the table, with the extra escaping needed for interpretation via sphinx.""")

    xmin = arrays.FloatArray(
        label="x-min",
        doc="""Minimum value""")

    xmax = arrays.FloatArray(
        label="x-max",
        doc="""Maximum value""")

    data = arrays.FloatArray(
        label="data",
        doc="""Tabulated values""")

    number_of_values = basic.Integer(
        label="Number of values",
        default=0,
        doc="""The number of values in the table """)

    df = arrays.FloatArray(
        label="df",
        doc=""".""")

    dx = arrays.FloatArray(
        label="dx",
        default=numpy.array([]),
        doc="""Tabulation step""")

    invdx = arrays.FloatArray(
        label="invdx",
        default=numpy.array([]),
        doc=""".""")



class PsiTableData(LookUpTableData):
    """
    Look up table containing the values of a function representing the time-averaged gating variable
    :math:`\\psi(\\nu)` as a function of the presynaptic rates :math:`\\nu` 
    
    """
    __tablename__ = None



class NerfTableData(LookUpTableData):
    """
    Look up table containing the values of Nerf integral within the :math:`\\phi`
    function that describes how the discharge rate vary as a function of parameters
    defining the statistical properties of the membrane potential in presence of synaptic inputs.
    
    """
    __tablename__ = None


