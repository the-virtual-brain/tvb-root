# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

import numpy
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import NArray, Int, Attr
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Json
from tvb.datatypes.surfaces import Surface

LOG = get_logger(__name__)

# Slices are for vertices [0.....SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE]
# [SPLIT_MAX_SIZE ..... 2 * SPLIT_BUFFER_SIZE + SPLIT_BUFFER_SIZE]
# Triangles [0, 1, 2], [3, 4, 5], [6, 7, 8].....
# Vertices -  no of triangles * 3

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


class SurfaceH5(H5File):

    def __init__(self, path):
        super(SurfaceH5, self).__init__(path)
        self.vertices = DataSet(Surface.vertices, self)
        self.triangles = DataSet(Surface.triangles, self)
        self.vertex_normals = DataSet(Surface.vertex_normals, self)
        self.triangle_normals = DataSet(Surface.triangle_normals, self)
        self.number_of_vertices = Scalar(Surface.number_of_vertices, self)
        self.number_of_triangles = Scalar(Surface.number_of_triangles, self)
        self.edge_mean_length = Scalar(Surface.edge_mean_length, self)
        self.edge_min_length = Scalar(Surface.edge_min_length, self)
        self.edge_max_length = Scalar(Surface.edge_max_length, self)
        self.zero_based_triangles = Scalar(Surface.zero_based_triangles, self)

        self.split_triangles = DataSet(NArray(dtype=int), self, name="split_triangles")
        self.number_of_split_slices = Scalar(Int(), self, name="number_of_split_slices")
        self.split_slices = Json(Attr(field_type=dict), self, name="split_slices")

        self.bi_hemispheric = Scalar(Surface.bi_hemispheric, self)
        self.surface_type = Scalar(Surface.surface_type, self)
        self.valid_for_simulations = Scalar(Surface.valid_for_simulations, self)

        # cached header like information, needed to interpret the rest of the file
        # Load the data that is required in order to interpret the file format
        # number_of_vertices and split_slices are needed for the get_vertices_slice read call

        if not self.is_new_file:
            self._split_slices = self.split_slices.load()
            self._split_triangles = self.split_triangles.load()
            self._number_of_vertices = self.number_of_vertices.load()
            self._number_of_triangles = self.number_of_triangles.load()
            self._number_of_split_slices = self.number_of_split_slices.load()
            self._bi_hemispheric = self.bi_hemispheric.load()
        # else: this is a new file

    def store(self, datatype, scalars_only=False, store_references=True):
        # type: (Surface, bool, bool) -> None
        super(SurfaceH5, self).store(datatype, scalars_only=scalars_only, store_references=store_references)
        # When any of the header fields change we have to update our cache of them
        # As they are an invariant of SurfaceH5 we don't do that in the accessors but here.
        # This implies that direct public writes to them via the accessors will break the invariant.
        # todo: should we make the accessors private? In complex formats like this one they are private
        # for this type direct writes to accessors should not be done
        self._number_of_vertices = datatype.number_of_vertices
        self._number_of_triangles = datatype.number_of_triangles
        self._bi_hemispheric = datatype.bi_hemispheric
        self.prepare_slices(datatype)
        self.number_of_split_slices.store(self._number_of_split_slices)
        self.split_slices.store(self._split_slices)
        self.split_triangles.store(self._split_triangles)

    def read_subtype_attr(self):
        return self.surface_type.load()

    def center(self):
        """
        Compute the center of the surface as the mean spot on all the three axes.
        """
        # is this different from return numpy.mean(self.vertices, axis=0) ?
        return [float(numpy.mean(self.vertices[:, 0])),
                float(numpy.mean(self.vertices[:, 1])),
                float(numpy.mean(self.vertices[:, 2]))]

    def get_number_of_split_slices(self):
        return self._number_of_split_slices

    def prepare_slices(self, datatype):
        """
        Before storing Surface in H5, make sure vertices/triangles are split in
        slices that are readable by WebGL.
        WebGL only supports triangle indices in interval [0.... 2^16]
        """
        # Do not split when size is conveniently small:
        if self._number_of_vertices <= SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE and not self._bi_hemispheric:
            self._number_of_split_slices = 1
            self._split_slices = {0: {KEY_TRIANGLES: {KEY_START: 0, KEY_END: self._number_of_triangles},
                                      KEY_VERTICES: {KEY_START: 0, KEY_END: self._number_of_vertices},
                                      KEY_HEMISPHERE: HEMISPHERE_UNKNOWN}}
            self._split_triangles = numpy.array([], dtype=numpy.int32)
            return

        # Compute the number of split slices:
        left_hemisphere_slices = 0
        left_hemisphere_vertices_no = 0
        if self._bi_hemispheric:
            # when more than one hemisphere
            right_hemisphere_vertices_no = numpy.count_nonzero(datatype.hemisphere_mask)
            left_hemisphere_vertices_no = self._number_of_vertices - right_hemisphere_vertices_no
            LOG.debug("Right %d Left %d" % (right_hemisphere_vertices_no, left_hemisphere_vertices_no))
            left_hemisphere_slices = self._get_slices_number(left_hemisphere_vertices_no)
            self._number_of_split_slices = left_hemisphere_slices
            self._number_of_split_slices += self._get_slices_number(right_hemisphere_vertices_no)
            LOG.debug("Hemispheres Total %d Left %d" % (self._number_of_split_slices, left_hemisphere_slices))
        else:
            # when a single hemisphere
            self._number_of_split_slices = self._get_slices_number(self._number_of_vertices)

        LOG.debug("Start to compute surface split triangles and vertices")
        split_triangles = []
        ignored_triangles_counter = 0
        self._split_slices = {}

        for i in range(self._number_of_split_slices):
            split_triangles.append([])
            if not self._bi_hemispheric:
                self._split_slices[i] = {KEY_VERTICES: {KEY_START: i * SPLIT_MAX_SIZE,
                                                        KEY_END: min(self._number_of_vertices,
                                                                     (i + 1) * SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE)},
                                         KEY_HEMISPHERE: HEMISPHERE_UNKNOWN}
            else:
                if i < left_hemisphere_slices:
                    self._split_slices[i] = {KEY_VERTICES: {KEY_START: i * SPLIT_MAX_SIZE,
                                                            KEY_END: min(left_hemisphere_vertices_no,
                                                                         (i + 1) * SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE)},
                                             KEY_HEMISPHERE: HEMISPHERE_LEFT}
                else:
                    self._split_slices[i] = {KEY_VERTICES: {KEY_START: left_hemisphere_vertices_no +
                                                                        (i - left_hemisphere_slices) * SPLIT_MAX_SIZE,
                                                            KEY_END: min(self._number_of_vertices,
                                                                         left_hemisphere_vertices_no + SPLIT_MAX_SIZE *
                                                                         (i + 1 - left_hemisphere_slices)
                                                                         + SPLIT_BUFFER_SIZE)},
                                             KEY_HEMISPHERE: HEMISPHERE_RIGHT}

        # Iterate Triangles and find the slice where it fits best, based on its vertices indexes:
        for i in range(self._number_of_triangles):
            current_triangle = [datatype.triangles[i][j] for j in range(3)]
            fit_slice, transformed_triangle = self._find_slice(current_triangle)

            if fit_slice is not None:
                split_triangles[fit_slice].append(transformed_triangle)
            else:
                # triangle ignored, as it has vertices over multiple slices.
                ignored_triangles_counter += 1
                continue

        final_split_triangles = []
        last_triangles_idx = 0

        # Concatenate triangles, to be stored in a single HDF5 array.
        for slice_idx, split_ in enumerate(split_triangles):
            self._split_slices[slice_idx][KEY_TRIANGLES] = {KEY_START: last_triangles_idx,
                                                            KEY_END: last_triangles_idx + len(split_)}
            final_split_triangles.extend(split_)
            last_triangles_idx += len(split_)
        self._split_triangles = numpy.array(final_split_triangles, dtype=numpy.int32)

        if ignored_triangles_counter > 0:
            LOG.warning("Ignored triangles from multiple hemispheres: " + str(ignored_triangles_counter))
        LOG.debug("End compute surface split triangles and vertices " + str(self._split_slices))

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
        mn = min(triangle)
        mx = max(triangle)
        for i in range(self._number_of_split_slices):
            v = self._split_slices[i][KEY_VERTICES]  # extracted for performance
            slice_start = v[KEY_START]
            if slice_start <= mn and mx < v[KEY_END]:
                return i, [triangle[j] - slice_start for j in range(3)]
        return None, triangle


    def get_slice_vertex_boundaries(self, slice_idx):
        if str(slice_idx) in self._split_slices:
            start_idx = max(0, self._split_slices[str(slice_idx)][KEY_VERTICES][KEY_START])
            end_idx = min(self._split_slices[str(slice_idx)][KEY_VERTICES][KEY_END], self._number_of_vertices)
            return start_idx, end_idx
        else:
            LOG.warning("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, min(SPLIT_BUFFER_SIZE, self._number_of_vertices)

    def _get_slice_triangle_boundaries(self, slice_idx):
        if str(slice_idx) in self._split_slices:
            start_idx = max(0, self._split_slices[str(slice_idx)][KEY_TRIANGLES][KEY_START])
            end_idx = min(self._split_slices[str(slice_idx)][KEY_TRIANGLES][KEY_END], self._number_of_triangles)
            return start_idx, end_idx
        else:
            LOG.warning("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, self._number_of_triangles

    def get_vertices_slice(self, slice_number=0):
        """
        Read vertices slice, to be used by WebGL visualizer.
        """
        slice_number = int(slice_number)
        start_idx, end_idx = self.get_slice_vertex_boundaries(slice_number)
        return self.vertices[start_idx: end_idx: 1]

    def get_vertex_normals_slice(self, slice_number=0):
        """
        Read vertex-normal slice, to be used by WebGL visualizer.
        """
        slice_number = int(slice_number)
        start_idx, end_idx = self.get_slice_vertex_boundaries(slice_number)
        return self.vertex_normals[start_idx: end_idx: 1]

    def get_triangles_slice(self, slice_number=0):
        """
        Read split-triangles slice, to be used by WebGL visualizer.
        """
        if self._number_of_split_slices == 1:
            return self.triangles.load()
        slice_number = int(slice_number)
        start_idx, end_idx = self._get_slice_triangle_boundaries(slice_number)
        return self._split_triangles[start_idx: end_idx: 1]

    def get_lines_slice(self, slice_number=0):
        """
        Read the gl lines values for the current slice number.
        """
        return Surface._triangles_to_lines(self.get_triangles_slice(slice_number))

    def get_slices_to_hemisphere_mask(self):
        """
        :return: a vector af length number_of_slices, with 1 when current chunk belongs to the Right hemisphere
        """
        if not self._bi_hemispheric or self._split_slices is None:
            return None
        result = [1] * self._number_of_split_slices
        for key, value in self._split_slices.items():
            if value[KEY_HEMISPHERE] == HEMISPHERE_LEFT:
                result[int(key)] = 0
        return result

    # todo: many of these do not belong in the data access layer but higher, adapter or gui layer
    ####################################### Split for Picking
    #######################################
    def get_pick_vertices_slice(self, slice_number=0):
        """
        Read vertices slice, to be used by WebGL visualizer with pick.
        """
        slice_number = int(slice_number)
        slice_triangles = self.triangles[
                          slice_number * SPLIT_PICK_MAX_TRIANGLE:
                          min(self._number_of_triangles, (slice_number + 1) * SPLIT_PICK_MAX_TRIANGLE)
                          ]
        result_vertices = []
        cache_vertices = self.vertices.load()
        for triang in slice_triangles:
            result_vertices.append(cache_vertices[triang[0]])
            result_vertices.append(cache_vertices[triang[1]])
            result_vertices.append(cache_vertices[triang[2]])
        return numpy.array(result_vertices)

    def get_pick_vertex_normals_slice(self, slice_number=0):
        """
        Read vertex-normals slice, to be used by WebGL visualizer with pick.
        """
        slice_number = int(slice_number)
        slice_triangles = self.triangles[
                          slice_number * SPLIT_PICK_MAX_TRIANGLE:
                          min(self.number_of_triangles.load(), (slice_number + 1) * SPLIT_PICK_MAX_TRIANGLE)
                          ]
        result_normals = []
        cache_vertex_normals = self.vertex_normals.load()

        for triang in slice_triangles:
            result_normals.append(cache_vertex_normals[triang[0]])
            result_normals.append(cache_vertex_normals[triang[1]])
            result_normals.append(cache_vertex_normals[triang[2]])
        return numpy.array(result_normals)

    def get_pick_triangles_slice(self, slice_number=0):
        """
        Read triangles slice, to be used by WebGL visualizer with pick.
        """
        slice_number = int(slice_number)
        no_of_triangles = (min(self._number_of_triangles, (slice_number + 1) * SPLIT_PICK_MAX_TRIANGLE)
                           - slice_number * SPLIT_PICK_MAX_TRIANGLE)
        triangles_array = numpy.arange(no_of_triangles * 3).reshape((no_of_triangles, 3))
        return triangles_array

# The h5 file format is not different for these surface subtypes
# so we should not create a new h5file
# If we will create a self.load -> Surface then that method should
# polymorphically decide on what surface subtype to construct
# class CorticalSurfaceH5(SurfaceH5):
