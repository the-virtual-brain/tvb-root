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
Scientific methods for the Pattern DataTypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import tvb.datatypes.patterns_data as patterns_data
import tvb.basic.traits.util as util
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class SpatioTemporalCall(object):
    """
    A call method to be added to all Spatio- Temporal classes
    """
    
    
    def __call__(self, temporal_indices=None, spatial_indices=None):
        """
        The temporal pattern vector, set by the configure_time method, is 
        combined with the spatial pattern vector, set by the configure_space 
        method, to form a spatiotemporal pattern.
        
        Called with a single time index as an argument, the spatial pattern at 
        that point in time is returned. This is the standard usage within a 
        simulation where the current simulation time point is retrieved.
        
        Called without any arguments, by default a big array representing the 
        entire spatio-temporal pattern is returned. While this may be useful for
        visualisation, say of region level spatio-temporal patterns, care should
        be taken as when surfaces are considered the returned array can be
        potentially quite large.
        """
        pattern = None
        if (temporal_indices is not None) and (spatial_indices is None):
            pattern = (self.spatial_pattern * self.temporal_pattern[0, temporal_indices])
        
        elif (temporal_indices is None) and (spatial_indices is None):
            pattern = self.spatial_pattern * self.temporal_pattern
        
        elif (temporal_indices is not None) and (spatial_indices is not None):
            pattern = (self.spatial_pattern[spatial_indices, 0] * self.temporal_pattern[0, temporal_indices])
        
        elif (temporal_indices is None) and (spatial_indices is not None):
            pattern = (self.spatial_pattern[spatial_indices, 0] * self.temporal_pattern)
        
        else:
            LOG.error("%s: Well, that shouldn't be possible..." % repr(self))
        return pattern



class SpatialPatternScientific(patterns_data.SpatialPatternData):
    """ This class exists to add scientific methods to SpatialPatternData. """
    space = None
    _spatial_pattern = None
    __tablename__ = None


    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this DataType.
        """
        return {"Type": self.__class__.__name__,
                "Spatial equation": self.spatial.__class__.__name__,
                "Spatial parameters": self.spatial.parameters}
    
    #--------------------------- spatial_pattern ------------------------------#
    def _get_spatial_pattern(self):
        """
        Return a discrete representation of the spatial pattern.
        """
        return self._spatial_pattern
        
    def _set_spatial_pattern(self, x):
        """ 
        Generate a discrete representation of the spatial pattern.
        The argument x represents a distance, or effective distance, for each node in the space.
        """
        self.spatial.pattern = x
        self._spatial_pattern = numpy.sum(self.spatial.pattern, axis=1)[:, numpy.newaxis]
    
    spatial_pattern = property(fget=_get_spatial_pattern, fset=_set_spatial_pattern)
    #--------------------------------------------------------------------------#


    def configure_space(self, distance):
        """
        Stores the distance vector as an attribute of the spatiotemporal pattern
        and uses it to generate the spatial pattern vector.
        
        Depending on equations used and interpretation distance can be an actual
        physical distance, on a surface,  geodesic distance (along the surface) 
        away for some focal point, or a per node weighting...
        """
        util.log_debug_array(LOG, distance, "distance")
        #Set the discrete representation of space.
        self.space = distance
        self.spatial_pattern = self.space



class SpatioTemporalPatternScientific(patterns_data.SpatioTemporalPatternData,
                                      SpatialPatternScientific, SpatioTemporalCall):
    """
    This class exists to add scientific methods to SpatioTemporalPatternData.
    """
    time = None
    _temporal_pattern = None
    __tablename__ = None
    
    
    def _find_summary_info(self):
        """ Extend the base class's summary dictionary. """
        summary = super(SpatioTemporalPatternScientific, self)._find_summary_info()
        summary["Temporal equation"] = self.temporal.__class__.__name__
        summary["Temporal parameters"] = self.temporal.parameters
        return summary


    #--------------------------- temporal_pattern -----------------------------#
    def _get_temporal_pattern(self):
        """
        Return a discrete representation of the temporal pattern.
        """
        return self._temporal_pattern


    def _set_temporal_pattern(self, t):
        """
        Generate a discrete representation of the temporal pattern.
        """
        self.temporal.pattern = t
        self._temporal_pattern = numpy.reshape(self.temporal.pattern, (1, -1))

    temporal_pattern = property(fget=_get_temporal_pattern, fset=_set_temporal_pattern)
    #--------------------------------------------------------------------------#


    def configure_time(self, time):
        """
        Stores the time vector, physical units (ms), as an attribute of the
        spatio-temporal pattern and uses it to generate the temporal pattern
        vector.
        """
        self.time = time
        self.temporal_pattern = self.time
    


class StimuliRegionScientific(patterns_data.StimuliRegionData, SpatioTemporalPatternScientific):
    """
    This class exists to add scientific methods to StimuliRegionData.
    """
    
    @property
    def weight_array(self):
        """
        Wrap weight List into a Numpy array, as it is requested by the simulator.
        """
        return numpy.array(self.weight)[:, numpy.newaxis]


    def configure_space(self, region_mapping=None):
        """
        Do necessary preparations in order to use this stimulus. 
        NOTE: this was previously done in simulator configure_stimuli() method.
        It no needs to be used in stimulus viewer also.
        """
        if region_mapping is not None:
            #TODO: smooth at surface region boundaries
            distance = self.weight_array[region_mapping, :]
        else:
            distance = self.weight_array
        super(StimuliRegionScientific, self).configure_space(distance)



class StimuliSurfaceScientific(patterns_data.StimuliSurfaceData, SpatioTemporalPatternScientific):
    """
    This class exists to add scientific methods to StimuliSurfaceData.
    """

    def configure_space(self, region_mapping=None):
        """
        Do necessary preparations in order to use this stimulus. 
        NOTE: this was previously done in simulator configure_stimuli() method.
        It no needs to be used in stimulus viewer also.
        """
        dis_shp = (self.surface.number_of_vertices, numpy.size(self.focal_points_surface))
        # TODO: When this was in Simulator it was number of nodes, using surface vertices
        # breaks surface simulations which include non-cortical regions.

        distance = numpy.zeros(dis_shp)
        k = -1
        for focal_point in self.focal_points_surface:
            k += 1
            foci = numpy.array([focal_point], dtype=numpy.int32)
            distance[:, k] = self.surface.geodesic_distance(foci)
        super(StimuliSurfaceScientific, self).configure_space(distance)



class SpatialPatternVolumeScientific(patterns_data.SpatialPatternVolumeData, SpatialPatternScientific):
    """
    This class exists to add scientific methods to SpatialPatternVolumeData.
    """
    pass

