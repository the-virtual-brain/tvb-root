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

Framework methods for the Pattern datatypes.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import tvb.datatypes.patterns_data as patterns_data


class SpatialPatternFramework(patterns_data.SpatialPatternData):
    """ This class exists to add framework methods to SpatialPatternData. """
    __tablename__ = None


class SpatioTemporalPatternFramework(patterns_data.SpatioTemporalPatternData):
    """
    This class exists to add framework methods to SpatioTemporalPatternData.
    """
    __tablename__ = None


class StimuliRegionFramework(patterns_data.StimuliRegionData):
    """ This class exists to add framework methods to StimuliRegionData. """
    __tablename__ = None

    @staticmethod
    def get_default_weights(number_of_regions):
        """
        Returns a list with a number of elements
        equal to the given number of regions.
        """
        default_weights = []
        for i in range(number_of_regions):
            default_weights.append(0.0)
        return default_weights


class StimuliSurfaceFramework(patterns_data.StimuliSurfaceData):
    """ This class exists to add framework methods to StimuliSurfaceData. """
    __tablename__ = None


class SpatialPatternVolumeFramework(patterns_data.SpatialPatternVolumeData):
    """
    This class exists to add framework methods to SpatialPatternVolumeData.
    """
    __tablename__ = None