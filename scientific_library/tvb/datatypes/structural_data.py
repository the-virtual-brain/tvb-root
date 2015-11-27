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
The Data component of Volumetric datatypes.

.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>

"""

import tvb.basic.traits.types_basic as basic
from tvb.datatypes import volumes
from tvb.datatypes.arrays import MappedArray, FloatArray


class StructuralMRIData(MappedArray):
    """
    Quantitative volumetric data recorded by means of Magnetic Resonance Imaging
    """

    # without the field below weighting and volume columns are going to be added to the MAPPED_ARRAY table
    __generate_table__ = True

    array_data = FloatArray(label= "contrast")

    weighting = basic.String(label= "MRI weighting") # eg, "T1", "T2", "T2*", "PD", ...

    volume = volumes.Volume

