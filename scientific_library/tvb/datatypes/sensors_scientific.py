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
Scientific methods for the Sensor dataTypes.

.. moduleauthor:: Lia Domide <lia@tvb.invalid>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import tvb.datatypes.sensors_data as sensors_data
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class SensorsScientific(sensors_data.SensorsData):
    """ This class exists to add scientific methods to SensorsData. """
    __tablename__ = None
    
    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """
        super(SensorsScientific, self).configure()
        self.number_of_sensors = self.labels.shape[0]

    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {"Sensor type": self.sensors_type,
                   "Number of Sensors": self.number_of_sensors}
        return summary
    
    
    def sensors_to_surface(self, surface_to_map):
        """
        Map EEG sensors onto the head surface (skin-air).
        
        EEG sensor locations are typically only given on a unit sphere, that is,
        they are effectively only identified by their orientation with respect
        to a coordinate system. This method is used to map these unit vector
        sensor "locations" to a specific location on the surface of the skin.
        
        Assumes coordinate systems are aligned, i.e. common x,y,z and origin.
        
        """
        #TODO: Think I did this assuming zero centered head, if so need to 
        #      change to support coordinate origins not in the center... Should
        #      just require a check with offset included in calculus though as it
        #      won't be center of mass, getting the correct offset will be a pain.
        #SK  : Works but it's a bit of a mess, cleanup...
        
        #Normalize sensor and vertex locations to unit vectors
        norm_sensors = numpy.sqrt(numpy.sum(self.locations ** 2, axis=1))
        unit_sensors = self.locations / norm_sensors[:, numpy.newaxis]
        norm_verts = numpy.sqrt(numpy.sum(surface_to_map.vertices ** 2, axis=1))
        unit_vertices = surface_to_map.vertices / norm_verts[:, numpy.newaxis]
        
        sensor_tri = numpy.zeros((self.number_of_sensors, 1), dtype=numpy.int32)
        sensor_locations = numpy.zeros((self.number_of_sensors, 3))
        for k in range(self.number_of_sensors):
            #Find the surface vertex most closely aligned with current sensor.
            sensor_loc = unit_sensors[k]
            allignment = numpy.dot(sensor_loc, unit_vertices.T)
            one_ring = []
            while not one_ring:
                closest_vertex = allignment.argmax()
                
                #Get the set of triangles in the neighbourhood of that vertex.
                #NOTE: Intersection doesn't always fall within the 1-ring, so, all 
                #      triangles contained in the 2-ring are considered.
                one_ring = surface_to_map.vertex_neighbours[closest_vertex]
                if not one_ring:
                    allignment[closest_vertex] = min(allignment)
            local_tri = [surface_to_map.vertex_triangles[vertex] for vertex in one_ring]
            local_tri = list(set([tri for subar in local_tri for tri in subar]))
            
            #Calculate a parameterized plane line intersection [t,u,v] for the
            #set of local triangles, which are considered as defining a plane.
            tuv = numpy.zeros((len(local_tri), 3))
            i = 0
            for tri in local_tri:
                edge_01 = (surface_to_map.vertices[surface_to_map.triangles[tri, 0]] - 
                           surface_to_map.vertices[surface_to_map.triangles[tri, 1]])
                edge_02 = (surface_to_map.vertices[surface_to_map.triangles[tri, 0]] - 
                           surface_to_map.vertices[surface_to_map.triangles[tri, 2]])
                see_mat = numpy.vstack((sensor_loc, edge_01, edge_02))
                
                tuv[i] = numpy.linalg.solve(see_mat.T, surface_to_map.vertices[surface_to_map.triangles[tri, 0].T])
                i += 1
            #import pdb; pdb.set_trace()
            #Find find which line-plane intersection falls within its triangle 
            #by imposing the condition that u, v, & u+v are contained in [0 1]
            local_triangle_index = ((0 < tuv[:, 1]) * (tuv[:, 1] < 1) *
                                    (0 < tuv[:, 2]) * (tuv[:, 2] < 1) *
                                    (0 < (tuv[:, 1] + tuv[:, 2])) *
                                    ((tuv[:, 1] + tuv[:, 2]) < 1)).nonzero()[0]
            #TODO: add checks for no or multiple intersections...
            #      no: surface incomplete, misaligned, irregular triangulation
            #      multiple: surface possibly too folded or misaligned...

            if len(local_triangle_index) > 0:
                sensor_tri[k] = local_tri[local_triangle_index[0]]
                #Scale sensor unit vector by t so that it lies on the surface.
                sensor_locations[k] = sensor_loc * tuv[local_triangle_index, 0]
            else:
                ## TODO: is this a wrong default?
                sensor_tri[k] = local_tri[0]
                sensor_locations[k] = self.locations[k]
                LOG.warning("Could not find a proper position on the given surface for sensor %d:%s. "
                            "It will appear in %s" % (k, self.labels[k], str(self.locations[k])))

        # sensor_tri seems to be unused
        # commented to make async data retrieval easy from the ui
        #return sensor_tri, sensor_locations
        return sensor_locations



class SensorsEEGScientific(sensors_data.SensorsEEGData, SensorsScientific):
    """ This class exists to add scientific methods to SensorsEEGData. """
    pass


class SensorsMEGScientific(sensors_data.SensorsMEGData, SensorsScientific):
    """ This class exists to add scientific methods to SensorsMEGData. """
    pass


class SensorsInternalScientific(sensors_data.SensorsInternalData,
                                SensorsScientific):
    """ This class exists to add scientific methods to SensorsInternalData. """
    pass

