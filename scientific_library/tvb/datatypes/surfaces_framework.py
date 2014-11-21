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
Framework methods for the Surface DataTypes.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
"""

import json
import numpy
import tvb.datatypes.surfaces_data as surfaces_data
import tvb.basic.traits.exceptions as exceptions
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)



# TODO: This is just a temporary solution placed here to remove dependency from tvb.basic to tvb framework.
# As soon as we implement a better solution to the datatype framework diamond problem this should be removed.
def paths2url(datatype_entity, attribute_name, flatten=False, parameter=None, datatype_kwargs=None):
    """
    Prepare a File System Path for passing into an URL.
    """
    if parameter is None:
        return (TvbProfile.current.web.VISUALIZERS_URL_PREFIX + datatype_entity.gid + '/' + attribute_name +
                '/' + str(flatten) + '/' + json.dumps(datatype_kwargs))

    return (TvbProfile.current.web.VISUALIZERS_URL_PREFIX + datatype_entity.gid + '/' + attribute_name +
            '/' + str(flatten) + '/' + json.dumps(datatype_kwargs) + "?" + str(parameter))


##--------------------- CLOSE SURFACES Start Here---------------------------------------##


#### Slices are for vertices [0.....SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE]
#### [SPLIT_MAX_SIZE ..... 2 * SPLIT_BUFFER_SIZE + SPLIT_BUFFER_SIZE]
#### Triangles [0, 1, 2], [3, 4, 5], [6, 7, 8].....
#### Vertices -  no of triangles * 3

SPLIT_MAX_SIZE = 40000
SPLIT_BUFFER_SIZE = 15000
SPLIT_PICK_MAX_TRIANGLE = 20000

KEY_TRIANGLES = "triangles"
KEY_VERTICES = "vertices"
KEY_HEMISPHERE = "hemisphere"
KEY_START = "start_idx"
KEY_END = "end_idx"

HEMISPHERE_LEFT = "LEFT"
HEMISPHERE_RIGHT = "RIGHT"
HEMISPHERE_UNKNOWN = "NONE"


class SurfaceFramework(surfaces_data.SurfaceData):
    """ 
    This class exists to add framework methods to SurfacesData.
    """
    __tablename__ = None

    def load_from_metadata(self, meta_dictionary):
        self.edge_mean_length = 0
        self.edge_min_length = 0
        self.edge_max_length = 0
        self.valid_for_simulations = True
        super(SurfaceFramework, self).load_from_metadata(meta_dictionary)

     
    def get_vertices_slice(self, slice_number=0):
        """
        Read vertices slice, to be used by WebGL visualizer.
        """
        slice_number = int(slice_number)
        start_idx, end_idx = self._get_slice_vertex_boundaries(slice_number)
        return self.get_data('vertices', slice(start_idx, end_idx, 1))
    
    
    def get_vertex_normals_slice(self, slice_number=0):
        """
        Read vertex-normal slice, to be used by WebGL visualizer.
        """
        slice_number = int(slice_number)
        start_idx, end_idx = self._get_slice_vertex_boundaries(slice_number)
        return self.get_data('vertex_normals', slice(start_idx, end_idx, 1))
    
    
    def get_triangles_slice(self, slice_number=0):
        """
        Read split-triangles slice, to be used by WebGL visualizer.
        """
        if self.number_of_split_slices == 1:
            return self.triangles
        slice_number = int(slice_number)
        start_idx, end_idx = self._get_slice_triangle_boundaries(slice_number)
        return self.get_data('split_triangles', slice(start_idx, end_idx, 1))
    
    
    def get_lines_slice(self, slice_number=0):
        """
        Read the gl lines values for the current slice number.
        """
        return self._triangles_to_lines(self.get_triangles_slice(slice_number))
    

    def get_slices_to_hemisphere_mask(self):
        """
        :return: a vector af length number_of_slices, with 1 when current chunk belongs to the Right hemisphere
        """
        if not self.bi_hemispheric or self.split_slices is None:
            return None
        result = [1] * self.number_of_split_slices
        for key, value in self.split_slices.iteritems():
            if value[KEY_HEMISPHERE] == HEMISPHERE_LEFT:
                result[int(key)] = 0
        return result


    @staticmethod
    def _triangles_to_lines(triangles):
        lines_array = []
        for a, b, c in triangles:
            lines_array.extend([a, b, b, c, c, a])
        return numpy.array(lines_array)

    
    def configure(self):
        """
        Before storing Surface in DB, make sure vertices/triangles are split in
        slices that are readable by WebGL.
        WebGL only supports triangle indices in interval [0.... 2^16] 
        """
        #super(SurfaceFramework, self).configure()

        self.number_of_vertices = self.vertices.shape[0]
        self.number_of_triangles = self.triangles.shape[0]

        ### Do not split again, if split-data is already computed:
        if 1 < self.number_of_split_slices == len(self.split_slices):
            return

        ### Do not split when size is conveniently small:
        self.bi_hemispheric = self.hemisphere_mask is not None and numpy.unique(self.hemisphere_mask).size > 1
        if self.number_of_vertices <= SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE and not self.bi_hemispheric:
            self.number_of_split_slices = 1
            self.split_slices = {0: {KEY_TRIANGLES: {KEY_START: 0, KEY_END: self.number_of_triangles},
                                     KEY_VERTICES: {KEY_START: 0, KEY_END: self.number_of_vertices},
                                     KEY_HEMISPHERE: HEMISPHERE_UNKNOWN}}
            return

        ### Compute the number of split slices:
        left_hemisphere_slices = 0
        left_hemisphere_vertices_no = 0
        if self.bi_hemispheric:
            ## when more than one hemisphere
            right_hemisphere_vertices_no = numpy.count_nonzero(self.hemisphere_mask)
            left_hemisphere_vertices_no = self.number_of_vertices - right_hemisphere_vertices_no
            LOG.debug("Right %d Left %d" % (right_hemisphere_vertices_no, left_hemisphere_vertices_no))
            left_hemisphere_slices = self._get_slices_number(left_hemisphere_vertices_no)
            self.number_of_split_slices = left_hemisphere_slices
            self.number_of_split_slices += self._get_slices_number(right_hemisphere_vertices_no)
            LOG.debug("Hemispheres Total %d Left %d" % (self.number_of_split_slices, left_hemisphere_slices))
        else:
            ## when a single hemisphere
            self.number_of_split_slices = self._get_slices_number(self.number_of_vertices)

        
        LOG.debug("Start to compute surface split triangles and vertices")
        split_triangles = []
        ignored_triangles_counter = 0
        self.split_slices = {}

        for i in xrange(self.number_of_split_slices):
            split_triangles.append([])
            if not self.bi_hemispheric:
                self.split_slices[i] = {KEY_VERTICES: {KEY_START: i * SPLIT_MAX_SIZE,
                                                       KEY_END: min(self.number_of_vertices,
                                                                    (i + 1) * SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE)},
                                        KEY_HEMISPHERE: HEMISPHERE_UNKNOWN}
            else:
                if i < left_hemisphere_slices:
                    self.split_slices[i] = {KEY_VERTICES: {KEY_START: i * SPLIT_MAX_SIZE,
                                                           KEY_END: min(left_hemisphere_vertices_no,
                                                                        (i + 1) * SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE)},
                                            KEY_HEMISPHERE: HEMISPHERE_LEFT}
                else:
                    self.split_slices[i] = {KEY_VERTICES: {KEY_START: left_hemisphere_vertices_no +
                                                                      (i - left_hemisphere_slices) * SPLIT_MAX_SIZE,
                                                           KEY_END: min(self.number_of_vertices,
                                                                        left_hemisphere_vertices_no + SPLIT_MAX_SIZE *
                                                                        (i + 1 - left_hemisphere_slices)
                                                                        + SPLIT_BUFFER_SIZE)},
                                            KEY_HEMISPHERE: HEMISPHERE_RIGHT}

        ### Iterate Triangles and find the slice where it fits best, based on its vertices indexes:
        for i in xrange(self.number_of_triangles):
            current_triangle = [self.triangles[i][j] for j in range(3)]
            fit_slice, transformed_triangle = self._find_slice(current_triangle)

            if fit_slice is not None:
                split_triangles[fit_slice].append(transformed_triangle)
            else:
                # triangle ignored, as it has vertices over multiple slices.
                ignored_triangles_counter += 1
                continue

        final_split_triangles = []
        last_triangles_idx = 0

        ### Concatenate triangles, to be stored in a single HDF5 array.
        for slice_idx, split_ in enumerate(split_triangles):
            self.split_slices[slice_idx][KEY_TRIANGLES] = {KEY_START: last_triangles_idx,
                                                           KEY_END: last_triangles_idx + len(split_)}
            final_split_triangles.extend(split_)
            last_triangles_idx += len(split_)
        self.split_triangles = numpy.array(final_split_triangles, dtype=numpy.int32)

        if ignored_triangles_counter > 0:
            LOG.warning("Ignored triangles from multiple hemispheres: " + str(ignored_triangles_counter))
        LOG.debug("End compute surface split triangles and vertices " + str(self.split_slices))
       

    def validate(self):
        # First check if the surface has a valid number of vertices
        self.number_of_vertices = self.vertices.shape[0]
        self.number_of_triangles = self.triangles.shape[0]

        if self.number_of_vertices > TvbProfile.current.MAX_SURFACE_VERTICES_NUMBER:
            msg = "This surface has too many vertices (max: %d)." % TvbProfile.current.MAX_SURFACE_VERTICES_NUMBER
            msg += " Please upload a new surface or change max number in application settings."
            raise exceptions.ValidationException(msg)
        return surfaces_data.ValidationResult()


    def get_urls_for_rendering(self, include_alphas=False, region_mapping=None): 
        """
        Compose URLs for the JS code to retrieve a surface from the UI for rendering.
        """
        url_vertices = []
        url_triangles = []
        url_normals = []
        url_lines = []
        alphas = []
        alphas_indices = []
        for i in xrange(self.number_of_split_slices):
            param = "slice_number=" + str(i)
            url_vertices.append(paths2url(self, 'get_vertices_slice', parameter=param, flatten=True))
            url_triangles.append(paths2url(self, 'get_triangles_slice', parameter=param, flatten=True))
            url_lines.append(paths2url(self, 'get_lines_slice', parameter=param, flatten=True))
            url_normals.append(paths2url(self, 'get_vertex_normals_slice', parameter=param, flatten=True))
            if not include_alphas or region_mapping is None:
                continue

            start_idx, end_idx = self._get_slice_vertex_boundaries(i)
            alphas.append(paths2url(region_mapping, "get_alpha_array", flatten=True,
                                    parameter="size=" + str(self.number_of_vertices)))
            alphas_indices.append(paths2url(region_mapping, "get_alpha_indices_array", flatten=True,
                                            parameter="start_idx=" + str(start_idx) + " ;end_idx=" + str(end_idx)))
          
        if include_alphas:  
            return url_vertices, url_normals, url_lines, url_triangles, alphas, alphas_indices
        return url_vertices, url_normals, url_lines, url_triangles


    def _get_slice_vertex_boundaries(self, slice_idx):
        if str(slice_idx) in self.split_slices:
            start_idx = max(0, self.split_slices[str(slice_idx)][KEY_VERTICES][KEY_START])
            end_idx = min(self.split_slices[str(slice_idx)][KEY_VERTICES][KEY_END], self.number_of_vertices)
            return start_idx, end_idx
        else:
            LOG.warn("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, min(SPLIT_BUFFER_SIZE, self.number_of_vertices)


    def _get_slice_triangle_boundaries(self, slice_idx):
        if str(slice_idx) in self.split_slices:
            start_idx = max(0, self.split_slices[str(slice_idx)][KEY_TRIANGLES][KEY_START])
            end_idx = min(self.split_slices[str(slice_idx)][KEY_TRIANGLES][KEY_END], self.number_of_triangles)
            return start_idx, end_idx
        else:
            LOG.warn("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, self.number_of_triangles


    @staticmethod
    def _get_slices_number(vertices_number):
        """
        Slices are for vertices [SPLIT_MAX_SIZE * i ... SPLIT_MAX_SIZE * (i + 1) + SPLIT_BUFFER_SIZE]
        Slices will overlap :
        |........SPLIT_MAX_SIZE|...SPLIT_BUFFER_SIZE|                           <-- split 1
                               |......... SPLIT_MAX_SIZE|...SPLIT_BUFFER_SIZE|  <-- split 2
        If we have trailing data smaller than the SPLIT_BUFFER_SIZE,
        then we no longer split but we need to have at least 1 slice.
        """
        slices_number, trailing = divmod(vertices_number, SPLIT_MAX_SIZE)
        if trailing > SPLIT_BUFFER_SIZE or (slices_number == 0 and trailing > 0):
            slices_number += 1
        return slices_number


    def _find_slice(self, triangle):
        split_slices = self.split_slices  # because of performance: 1.5 times slower without this
        mn = min(triangle)
        mx = max(triangle)
        for i in xrange(self.number_of_split_slices):
            v = split_slices[i][KEY_VERTICES]   # extracted for performance
            slice_start = v[KEY_START]
            if slice_start <= mn and mx < v[KEY_END]:
                return i, [triangle[j] - slice_start for j in range(3)]
        return None, triangle

    ####################################### Split for Picking
    #######################################
    def get_pick_vertices_slice(self, slice_number=0):
        """
        Read vertices slice, to be used by WebGL visualizer with pick.
        """
        slice_number = int(slice_number)
        slice_triangles = self.get_data('triangles', slice(slice_number * SPLIT_PICK_MAX_TRIANGLE,
                                                           min(self.number_of_triangles,
                                                               (slice_number + 1) * SPLIT_PICK_MAX_TRIANGLE)))
        result_vertices = []
        for triang in slice_triangles:
            result_vertices.append(self.vertices[triang[0]])
            result_vertices.append(self.vertices[triang[1]])
            result_vertices.append(self.vertices[triang[2]])
        return numpy.array(result_vertices)
       
       
    def get_pick_vertex_normals_slice(self, slice_number=0):
        """
        Read vertex-normals slice, to be used by WebGL visualizer with pick.
        """
        slice_number = int(slice_number)
        slice_triangles = self.get_data('triangles', slice(slice_number * SPLIT_PICK_MAX_TRIANGLE,
                                                           min(self.number_of_triangles,
                                                               (slice_number + 1) * SPLIT_PICK_MAX_TRIANGLE)))
        result_normals = []
        for triang in slice_triangles:
            result_normals.append(self.vertex_normals[triang[0]])
            result_normals.append(self.vertex_normals[triang[1]])
            result_normals.append(self.vertex_normals[triang[2]])
        return numpy.array(result_normals) 
         
         
    def get_pick_triangles_slice(self, slice_number=0):
        """
        Read triangles slice, to be used by WebGL visualizer with pick.
        """
        slice_number = int(slice_number)
        no_of_triangles = (min(self.number_of_triangles, (slice_number + 1) * SPLIT_PICK_MAX_TRIANGLE)
                           - slice_number * SPLIT_PICK_MAX_TRIANGLE)
        triangles_array = numpy.arange(no_of_triangles * 3).reshape((no_of_triangles, 3))
        return triangles_array
            
         
    def get_urls_for_pick_rendering(self):
        """
        Compose URLS for the JS code to retrieve a surface for picking.
        """
        vertices = []
        triangles = []
        normals = []
        number_of_split = self.number_of_triangles // SPLIT_PICK_MAX_TRIANGLE
        if self.number_of_triangles % SPLIT_PICK_MAX_TRIANGLE > 0:
            number_of_split += 1
            
        for i in xrange(number_of_split):
            param = "slice_number=" + str(i)
            vertices.append(paths2url(self, 'get_pick_vertices_slice', parameter=param, flatten=True))
            triangles.append(paths2url(self, 'get_pick_triangles_slice', parameter=param, flatten=True))
            normals.append(paths2url(self, 'get_pick_vertex_normals_slice', parameter=param, flatten=True))
            
        return vertices, normals, triangles
    
    
    def get_url_for_region_boundaries(self, region_mapping):
        return paths2url(self, 'generate_region_boundaries', datatype_kwargs={'region_mapping': region_mapping.gid})
    
    
    def center(self):
        """
        Compute the center of the surface as the mean spot on all the three axes.
        """
        # is this different from return numpy.mean(self.vertices, axis=0) ?
        return [float(numpy.mean(self.vertices[:, 0])),
                float(numpy.mean(self.vertices[:, 1])),
                float(numpy.mean(self.vertices[:, 2]))]


    def generate_region_boundaries(self, region_mapping):
        """
        Return the full region boundaries, including: vertices, normals and lines indices.
        """
        boundary_vertices = []
        boundary_lines = []
        boundary_normals = []
        array_data = region_mapping.array_data

        for slice_idx in xrange(self.number_of_split_slices):
            # Generate the boundaries sliced for the off case where we might overflow the buffer capacity
            slice_triangles = self.get_triangles_slice(slice_idx)
            slice_vertices = self.get_vertices_slice(slice_idx)
            slice_normals = self.get_vertex_normals_slice(slice_idx)
            first_index_in_slice = self.split_slices[str(slice_idx)][KEY_VERTICES][KEY_START]
            # These will keep track of the vertices / triangles / normals for this slice that have
            # been processed and were found as a part of the boundary
            processed_vertices = []
            processed_triangles = []
            processed_normals = []
            for triangle in slice_triangles:
                triangle += first_index_in_slice
                # Check if there are two points from a triangles that are in separate regions
                # then send this to further processing that will generate the corresponding
                # region separation lines depending on the 3rd point from the triangle
                rt0, rt1, rt2 = array_data[triangle]
                if rt0 - rt1:
                    reg_idx1, reg_idx2, dangling_idx = 0, 1, 2
                elif rt1 - rt2:
                    reg_idx1, reg_idx2, dangling_idx = 1, 2, 0
                elif rt2 - rt0:
                    reg_idx1, reg_idx2, dangling_idx = 2, 0, 1
                else:
                    continue

                lines_vert, lines_ind, lines_norm = self._process_triangle(triangle, reg_idx1, reg_idx2, dangling_idx,
                                                                           first_index_in_slice, array_data,
                                                                           slice_vertices, slice_normals)
                ind_offset = len(processed_vertices) / 3
                processed_vertices.extend(lines_vert)
                processed_normals.extend(lines_norm)
                processed_triangles.extend([ind + ind_offset for ind in lines_ind])
            boundary_vertices.append(processed_vertices)
            boundary_lines.append(processed_triangles)
            boundary_normals.append(processed_normals)
        return numpy.array([boundary_vertices, boundary_lines, boundary_normals])
                    
                    
            
    @staticmethod
    def _process_triangle(triangle, reg_idx1, reg_idx2, dangling_idx, indices_offset,
                          region_mapping_array, vertices, normals):
        """
        Process a triangle and generate the required data for a region separation.
        :param triangle: the actual triangle as a 3 element vector
        :param reg_idx1: the first vertex that is in a 'conflicting' region
        :param reg_idx2: the second vertex that is in a 'conflicting' region
        :param dangling_idx: the third vector for which we know nothing yet.
                    Depending on this we might generate a line, or a 3 star centered in the triangle
        :param indices_offset: to take into account the slicing
        :param region_mapping_array: the region mapping raw array for which the regions are computed
        :param vertices: the current vertex slice
        :param normals: the current normals slice
        """
        def _star_triangle(point0, point1, point2, result_array):
            """
            Helper function that for a given triangle generates a 3-way star centered in the triangle center
            """
            center_vertex = [(point0[i] + point1[i] + point2[i]) / 3 for i in xrange(3)]
            mid_line1 = [(point0[i] + point1[i]) / 2 for i in xrange(3)]
            mid_line2 = [(point1[i] + point2[i]) / 2 for i in xrange(3)]
            mid_line3 = [(point2[i] + point0[i]) / 2 for i in xrange(3)]
            result_array.extend(center_vertex)
            result_array.extend(mid_line1)
            result_array.extend(mid_line2)
            result_array.extend(mid_line3)
            
        def _slice_triangle(point0, point1, point2, result_array):
            """
            Helper function that for a given triangle generates a line cutting thtough the middle of two edges.
            """
            mid_line1 = [(point0[i] + point1[i]) / 2 for i in xrange(3)]
            mid_line2 = [(point0[i] + point2[i]) / 2 for i in xrange(3)]
            result_array.extend(mid_line1)
            result_array.extend(mid_line2)

        # performance opportunity: we are computing some values available in caller

        p0 = vertices[triangle[reg_idx1] - indices_offset]
        p1 = vertices[triangle[reg_idx2] - indices_offset]
        p2 = vertices[triangle[dangling_idx] - indices_offset]
        n0 = normals[triangle[reg_idx1] - indices_offset]
        n1 = normals[triangle[reg_idx2] - indices_offset]
        n2 = normals[triangle[dangling_idx] - indices_offset]
        result_vertices = []
        result_normals = []

        dangling_reg = region_mapping_array[triangle[dangling_idx]]
        reg_1 = region_mapping_array[triangle[reg_idx1]]
        reg_2 = region_mapping_array[triangle[reg_idx2]]

        if dangling_reg != reg_1 and dangling_reg != reg_2:
            # Triangle is actually spanning 3 regions. Create a vertex in the center of the triangle, which connects to
            # the middle of each edge
            _star_triangle(p0, p1, p2, result_vertices)
            _star_triangle(n0, n1, n2, result_normals)
            result_lines = [0, 1, 0, 2, 0, 3]
        elif dangling_reg == reg_1:
            # Triangle spanning only 2 regions, draw a line through the middle of the triangle
            _slice_triangle(p1, p0, p2, result_vertices)
            _slice_triangle(n1, n0, n2, result_normals)
            result_lines = [0, 1]
        else:
            # Triangle spanning only 2 regions, draw a line through the middle of the triangle
            _slice_triangle(p0, p1, p2, result_vertices)
            _slice_triangle(n0, n1, n2, result_normals)
            result_lines = [0, 1]
        return result_vertices, result_lines, result_normals
    
        

class CorticalSurfaceFramework(surfaces_data.CorticalSurfaceData, SurfaceFramework):
    """ This class exists to add framework methods to CorticalSurfaceData """
    pass


class SkinAirFramework(surfaces_data.SkinAirData, SurfaceFramework):
    """ This class exists to add framework methods to SkinAirData """
    __tablename__ = None


class BrainSkullFramework(surfaces_data.BrainSkullData, SurfaceFramework):
    """ This class exists to add framework methods to BrainSkullData """
    pass


class SkullSkinFramework(surfaces_data.SkullSkinData, SurfaceFramework):
    """ This class exists to add framework methods to SkullSkinData """
    pass

##--------------------- CLOSE SURFACES End Here---------------------------------------##

##--------------------- OPEN SURFACES Start Here---------------------------------------##


class OpenSurfaceFramework(surfaces_data.OpenSurfaceData, SurfaceFramework):
    """ This class exists to add framework methods to OpenSurfaceData """



class EEGCapFramework(surfaces_data.EEGCapData, OpenSurfaceFramework):
    """ This class exists to add framework methods to EEGCapData """
    pass


class FaceSurfaceFramework(surfaces_data.FaceSurfaceData, OpenSurfaceFramework):
    """ This class exists to add framework methods to FaceSurface """
    pass

##--------------------- OPEN SURFACES End Here---------------------------------------##

##--------------------- SURFACES ADJIACENT classes start Here---------------------------------------##


class RegionMappingFramework(surfaces_data.RegionMappingData):
    """ 
    Framework methods regarding RegionMapping DataType.
    """
    __tablename__ = None
    
       
    @staticmethod
    def get_alpha_array(size):
        """
        Compute alpha weights.
        When displaying region-based results, we need to compute color for each
        surface vertex based on a gradient of the neighbor region(s).
        Currently only one vertex is used for determining color (the one 
        indicated by the RegionMapping).
        :return: NumPy array with [[1, 0], [1, 0] ....] of length :param size
        """
        if isinstance(size, (str, unicode)):
            size = int(size)
        return numpy.ones((size, 1)) * numpy.array([1.0, 0.0])
    
    
    def get_alpha_indices_array(self, start_idx, end_idx):
        """
        Compute alpha indices.
        When displaying region-based results, we need to compute color for each
        surface vertex based on a gradient of the neighbor region(s).
        For each vertex on the surface, alpha-indices will be the closest
        region-indices

        :param start_idx: vertex index on the surface
        :param end_idx: vertex index on the surface
        :return: NumPy array with [[colosest_reg_idx, 0, 0] ....]

        """
        if isinstance(start_idx, (str, unicode)):
            start_idx = int(start_idx)
        if isinstance(end_idx, (str, unicode)):
            end_idx = int(end_idx)
        size = end_idx - start_idx
        result = numpy.transpose(self.array_data[start_idx: end_idx]).reshape(size, 1) * numpy.array([1.0, 0.0, 0.0])
        result = result + numpy.ones((size, 1)) * numpy.array([0.0, 1.0, 1.0])
        return result
    
    
    def generate_new_region_mapping(self, connectivity_gid, storage_path):
        """
        Generate a new region mapping with the given connectivity gid from an 
        existing mapping corresponding to the parent connectivity.
        """
        new_region_map = self.__class__()
        new_region_map.storage_path = storage_path
        new_region_map._connectivity = connectivity_gid
        new_region_map._surface = self._surface
        new_region_map.array_data = self.array_data
        return new_region_map
    
    
class LocalConnectivityFramework(surfaces_data.LocalConnectivityData):
    """ This class exists to add framework methods to LocalConnectivityData """
    __tablename__ = None
    
    
class CortexFramework(surfaces_data.CortexData, CorticalSurfaceFramework):
    """ This class exists to add framework methods to CortexData """
    __tablename__ = None
    
    
