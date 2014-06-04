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
The Data component of Volumes datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

from tvb.basic.traits.types_mapped import MappedType
import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays


class VolumeData(MappedType):
    """
    Data having voxels as their elementary units.
    """
    origin = arrays.PositionArray(label = "Volume origin coordinates")
    voxel_size = arrays.FloatArray(label = "Voxel size") # need a triplet, xyz
    voxel_unit = basic.String(label = "Voxel Measure Unit", default = "mm")


class ParcellationMaskData(VolumeData):
    """
    This mask provides the information to perform a subdivision (parcellation) 
    of the brain `Volume` of the desired subject into spatially compacts 
    clusters or parcels. 
    This subdivision is based on spatial coordinates and functional information, 
    in order to grant spatially consistent and functionally homogeneous units.
    """
    data = arrays.IndexArray(label = "Parcellation mask")
    region_labels = arrays.StringArray(label = "Region labels")


class StructuralMRIData(VolumeData):
    """
    Quantitative volumetric data recorded by means of Magnetic Resonance Imaging 
    """
    #TODO: Need data defined ?data = arrays.FloatArray(label = "?Contrast?") ?
    weighting = basic.String(label = "MRI weighting") # eg, "T1", "T2", "T2*", "PD", ...
