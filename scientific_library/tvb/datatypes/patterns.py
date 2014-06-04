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

The Pattern datatypes. This brings together the scientific and framework
methods that are associated with the pattern datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import tvb.datatypes.patterns_scientific as patterns_scientific
import tvb.datatypes.patterns_framework as patterns_framework
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class SpatialPattern(patterns_scientific.SpatialPatternScientific,
                     patterns_framework.SpatialPatternFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the SpatialPattern datatype.
    
    ::
        
                         SpatialPatternData
                                 |
                                / \\
         SpatialPatternFramework   SpatialPatternScientific
                                \ /
                                 |
                          SpatialPattern
        
    
    """
    pass


class SpatioTemporalPattern(patterns_scientific.SpatioTemporalPatternScientific,
                            patterns_framework.SpatioTemporalPatternFramework, SpatialPattern):
    """
    This class brings together the scientific and framework methods that are
    associated with the SpatioTemporalPattern datatype.
    
    ::
        
                           SpatioTemporalPatternData
                                       |
                                      / \\
        SpatioTemporalPatternFramework   SpatioTemporalPatternScientific
                                      \ /
                                       |
                             SpatioTemporalPattern
        
    
    """


class StimuliRegion(patterns_scientific.StimuliRegionScientific,
                    patterns_framework.StimuliRegionFramework, SpatioTemporalPattern):
    """
    This class brings together the scientific and framework methods that are
    associated with the StimuliRegion datatype.
    
    ::
        
                          StimuliRegionData
                                 |
                                / \\
          StimuliRegionFramework   StimuliRegionScientific
                                \ /
                                 |
                           StimuliRegion
        
    
    """


class StimuliSurface(patterns_scientific.StimuliSurfaceScientific,
                     patterns_framework.StimuliSurfaceFramework, SpatioTemporalPattern):
    """
    This class brings together the scientific and framework methods that are
    associated with the StimuliSurface datatype.
    
    ::
        
                         StimuliSurfaceData
                                 |
                                / \\
         StimuliSurfaceFramework   StimuliSurfaceScientific
                                \ /
                                 |
                          StimuliSurface
        
    
    """


class SpatialPatternVolume(patterns_scientific.SpatialPatternVolumeScientific,
                           patterns_framework.SpatialPatternVolumeFramework, SpatialPattern):
    """
    This class brings together the scientific and framework methods that are
    associated with the SpatialPatternVolume datatype.
    
    ::
        
                            SpatialPatternVolumeData
                                       |
                                      / \\
         SpatialPatternVolumeFramework   SpatialPatternVolumeScientific
                                      \ /
                                       |
                              SpatialPatternVolume
        
    
    """
