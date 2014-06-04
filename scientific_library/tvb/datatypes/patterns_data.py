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
The Data component of Spatiotemporal pattern datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
"""


import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays
import tvb.datatypes.surfaces as surfaces
import tvb.datatypes.volumes as volumes
import tvb.datatypes.connectivity as connectivity_module
import tvb.datatypes.equations as equations
from tvb.basic.traits.types_mapped import MappedType



class SpatialPatternData(MappedType):
    """
    Equation for space variation.
    """

    spatial = equations.FiniteSupportEquation(label="Spatial Equation", order=2)



class SpatioTemporalPatternData(SpatialPatternData):
    """
    Combine space and time equations.
    """

    temporal = equations.TemporalApplicableEquation(label="Temporal Equation", order=3)
    #space must be shape (x, 1); time must be shape (1, t)




class StimuliRegionData(SpatioTemporalPatternData):
    """ 
    A class that bundles the temporal profile of the stimulus, together with the 
    list of scaling weights of the regions where it will applied.
    """

    connectivity = connectivity_module.Connectivity(label="Connectivity", order=1)

    spatial = equations.DiscreteEquation(label="Spatial Equation", default=equations.DiscreteEquation,
                                         fixed_type=True, order=-1)

    weight = basic.List(label="scaling", locked=True, order=4)



class StimuliSurfaceData(SpatioTemporalPatternData):
    """
    A spatio-temporal pattern defined in a Surface DataType.
    It includes the list of focal points.
    """

    surface = surfaces.CorticalSurface(label="Surface", order=1)

    focal_points_surface = basic.List(label="Focal points", locked=True, order=4)

    focal_points_triangles = basic.List(label="Focal points triangles", locked=True, order=4)



class SpatialPatternVolumeData(SpatialPatternData):
    """ A spatio-temporal pattern defined in a volume. """

    volume = volumes.Volume(label="Volume")

    focal_points_volume = arrays.IndexArray(label="Focal points", target=volume)


