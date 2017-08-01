# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
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

The RegionBoundaries class, given a cortical surface with region_mapping based
on a particular parcellation, provides access to region neighbours and 
boundaries within the surface.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
#TODO: debug, clean, doc
#
import numpy

#The Virtual Brain

from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

#import tvb.basic.traits as trait
#import tvb.simulator.surfaces as surfaces_module


class RegionBoundaries(object):
    """
    """
#TODO: Cortex attribute isn't properly initialising -- doing something wrong...    
#    cortex = trait.Instance(label = "A cortical surface object",
#                            default = surfaces_module.Cortex(),
#                            default_class = surfaces_module.Cortex,
#                            value = surfaces_module.Cortex(),  #trait_type
#                            datatype = True,
#                            required = True)

    def __init__(self, cortex): # **kwargs
        """
        """

#        super(RegionBoundaries, self).__init__(**kwargs)
        self.cortex = cortex
        self._boundary_triangles = None # All triangles that cross a boundary
        self._boundary_edges = None # All edges that cross a boundary
        self._neighbouring_regions = None
        self._region_neighbours = None # Dict of region neihbours
        self._region_adjacency= None # Adjacency matrix indicating region neighbours
        self._region_boundaries = None # Vertices at the edge of each region
        self._boundary = None


    @property
    def boundary_triangles(self):
        """ The ... """
        if self._boundary_triangles is None:
           self._boundary_triangles = find_boundary_triangles(self.cortex)
        return self._boundary_triangles


    @property
    def boundary_edges(self):
        """ The ... """
        if self._boundary_edges is None:
           self._boundary_edges = find_boundary_edges(self.cortex)
        return self._boundary_edges


    @property
    def neighbouring_regions(self):
        """ The ... """
        if self._neighbouring_regions is None:
           self._neighbouring_regions = find_neighbouring_regions(self.cortex, self.boundary_edges)
        return self._neighbouring_regions


    @property
    def region_neighbours(self):
        """ The ... """
        if self._region_neighbours is None:
           self._region_neighbours = find_region_neighbours(self.region_adjacency)
        return self._region_neighbours


    @property
    def number_of_neighbours(self):
        """ The ... """
        if self._number_of_neighbours is None:
           self._number_of_neighbours = find_number_of_neighbours(self.region_neighbours)
        return self._number_of_neighbours


    @property
    def region_adjacency(self):
        """ The ... """
        if self._region_adjacency is None:
           self._region_adjacency = find_region_adjacency(self.neighbouring_regions)
        return self._region_adjacency


    @property
    def region_boundaries(self):
        """ The ... """
        if self._region_boundaries is None:
           self._region_boundaries = find_region_boundaries(self.cortex)
        return self._region_boundaries


    @property
    def boundary(self):
        """ The ... """
        if self._boundary is None:
           self._boundary = find_boundary(self.cortex, self.boundary_edges)
        return self._boundary



def find_boundary_triangles(cortex):
    """ 
    Identify triangles that cross a parcellation boundary 
    """
    tb01 = numpy.nonzero(cortex.region_mapping[cortex.triangles][:, 0] - 
                         cortex.region_mapping[cortex.triangles][:, 1])

    tb12 = numpy.nonzero(cortex.region_mapping[cortex.triangles][:, 1] - 
                         cortex.region_mapping[cortex.triangles][:, 2])

    tb20 = numpy.nonzero(cortex.region_mapping[cortex.triangles][:, 2] - 
                         cortex.region_mapping[cortex.triangles][:, 0])

    return numpy.unique(numpy.hstack((tb01, tb12, tb20)))


def find_boundary_edges(cortex):
    """ 
    Identify edges that cross a parcellation boundary 
    """
    tb01 = numpy.nonzero(cortex.region_mapping[cortex.triangles][:, 0] -
                         cortex.region_mapping[cortex.triangles][:, 1])

    tb12 = numpy.nonzero(cortex.region_mapping[cortex.triangles][:, 1] -
                         cortex.region_mapping[cortex.triangles][:, 2])

    tb20 = numpy.nonzero(cortex.region_mapping[cortex.triangles][:, 2] -
                         cortex.region_mapping[cortex.triangles][:, 0])
    ed01 = numpy.vstack((cortex.triangles[tb01, 0], cortex.triangles[tb01, 1]))
    ed12 = numpy.vstack((cortex.triangles[tb12, 1], cortex.triangles[tb12, 2]))
    ed20 = numpy.vstack((cortex.triangles[tb20, 2], cortex.triangles[tb20, 0]))

    all_boundary_edges = numpy.hstack((ed01, ed12, ed20)).T
    all_boundary_edges.sort()
    boundary_edge_set = set(map(tuple, all_boundary_edges.tolist()))
    unique_boundary_edges = numpy.zeros((2, len(boundary_edge_set)), dtype=numpy.int32)
    k =0
    while boundary_edge_set:
        unique_boundary_edges[:, k] = numpy.array(boundary_edge_set.pop())
        k+=1
    return unique_boundary_edges


def find_region_boundaries(cortex, boundary_triangles):
    """ """
    boundary_vertices = numpy.unique(cortex.triangles[boundary_triangles, :])
    vertex_regions = cortex.region_mapping[boundary_vertices]
    region_boundaries = []
    for k in numpy.unique(cortex.region_mapping):
        region_boundaries.append(boundary_vertices[vertex_regions==k])
    return region_boundaries


def find_neighbouring_regions(cortex, boundary_edges):
    """ """
    nr = cortex.region_mapping[boundary_edges]
    nr = nr.T
    nr.sort()
    neighbouring_region_set = set(map(tuple, nr.tolist()))
    unique_neighbouring_regions = numpy.zeros((2, len(neighbouring_region_set)), dtype=numpy.int32)
    k = 0
    while neighbouring_region_set:
        unique_neighbouring_regions[:, k] = numpy.array(neighbouring_region_set.pop())
        k+=1
    return unique_neighbouring_regions


def find_boundary(cortex, boundary_edges):
    """
    """
    edge_vert_pairs = cortex.vertices[boundary_edges, :]
    vec_0to1 = (edge_vert_pairs[1, :] - edge_vert_pairs[0, :])
    boundary = edge_vert_pairs[0, :] +  vec_0to1 / 2.0
    return boundary


def find_region_adjacency(neighbouring_regions):
    """
    """
    number_of_regions = neighbouring_regions.max() + 1
    region_adjacency = numpy.zeros((number_of_regions, number_of_regions), 
                                   dtype=numpy.int32)

    region_adjacency[neighbouring_regions[0], neighbouring_regions[1]] = 1

    return region_adjacency



def find_region_neighbours(region_adjacency):
    """
    """
    #NOTE: 'should only be doing 1 hemisphere here and then flipping, for symmetry.
    number_of_regions = region_adjacency.shape[0]
    xxx = numpy.nonzero(region_adjacency + region_adjacency.T)
    neighbours = {}
    for key in range(number_of_regions):
        #Assign 
        neighbours[key] = list(xxx[1][xxx[0]==key])
#        # Extend neighbours to include "colourbar neighbours"
#        neighbours[key].append(numpy.mod(key+1, number_of_regions))
#        neighbours[key].append(numpy.mod(key-1, number_of_regions))
    return neighbours



def find_number_of_neighbours(neighbours):
    """
    """
    number_of_regions = len(neighbours)
    number_of_neighbours = numpy.zeros(number_of_regions, dtype=numpy.uint8)
    for k in range(number_of_regions):
        number_of_neighbours[k] = len(neighbours[k])


### EoF ###
