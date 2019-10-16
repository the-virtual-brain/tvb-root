import logging
import numpy
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Json
from tvb.datatypes.surfaces import Surface, KEY_VERTICES, KEY_START, KEY_END, SPLIT_BUFFER_SIZE, KEY_TRIANGLES, \
    KEY_HEMISPHERE, HEMISPHERE_LEFT, SPLIT_PICK_MAX_TRIANGLE

log = logging.getLogger(__name__)


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
        self.split_triangles = DataSet(Surface.split_triangles, self)
        self.number_of_split_slices = Scalar(Surface.number_of_split_slices, self)
        self.split_slices = Json(Surface.split_slices, self)
        self.bi_hemispheric = Scalar(Surface.bi_hemispheric, self)
        self.surface_type = Scalar(Surface.surface_type, self)
        self.valid_for_simulations = Scalar(Surface.valid_for_simulations, self)

        # cached header like information, needed to interpret the rest of the file
        # Load the data that is required in order to interpret the file format
        # number_of_vertices and split_slices are needed for the get_vertices_slice read call

        if self.storage_manager.is_valid_hdf5_file():
            self._split_slices = self.split_slices.load()
            self._number_of_vertices = self.number_of_vertices.load()
            self._number_of_triangles = self.number_of_triangles.load()
            self._number_of_split_slices = self.number_of_split_slices.load()
            self._bi_hemispheric = self.bi_hemispheric.load()
        # else: this is a new file


    def store(self, datatype, scalars_only=False):
        # type: (Surface, bool) -> None
        super(SurfaceH5, self).store(datatype, scalars_only=scalars_only)
        # When any of the header fields change we have to update our cache of them
        # As they are an invariant of SurfaceH5 we don't do that in the accessors but here.
        # This implies that direct public writes to them via the accessors will break the invariant.
        # todo: should we make the accessors private? In complex formats like this one they are private
        # for this type direct writes to accessors should not be done
        self._split_slices = datatype.split_slices
        self._number_of_vertices = datatype.number_of_vertices
        self._number_of_triangles = datatype.number_of_triangles
        self._number_of_split_slices = datatype.number_of_split_slices
        self._bi_hemispheric = datatype.bi_hemispheric


    # experimental port of some of the data access apis from the datatype


    def _get_slice_vertex_boundaries(self, slice_idx):
        if str(slice_idx) in self._split_slices:
            start_idx = max(0, self._split_slices[str(slice_idx)][KEY_VERTICES][KEY_START])
            end_idx = min(self._split_slices[str(slice_idx)][KEY_VERTICES][KEY_END], self._number_of_vertices)
            return start_idx, end_idx
        else:
            log.warning("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, min(SPLIT_BUFFER_SIZE, self._number_of_vertices)


    def _get_slice_triangle_boundaries(self, slice_idx):
        if str(slice_idx) in self._split_slices:
            start_idx = max(0, self._split_slices[str(slice_idx)][KEY_TRIANGLES][KEY_START])
            end_idx = min(self._split_slices[str(slice_idx)][KEY_TRIANGLES][KEY_END], self._number_of_triangles)
            return start_idx, end_idx
        else:
            log.warn("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, self._number_of_triangles


    def get_vertices_slice(self, slice_number=0):
        """
        Read vertices slice, to be used by WebGL visualizer.
        """
        slice_number = int(slice_number)
        start_idx, end_idx = self._get_slice_vertex_boundaries(slice_number)
        return self.vertices[start_idx: end_idx: 1]


    def get_vertex_normals_slice(self, slice_number=0):
        """
        Read vertex-normal slice, to be used by WebGL visualizer.
        """
        slice_number = int(slice_number)
        start_idx, end_idx = self._get_slice_vertex_boundaries(slice_number)
        return self.vertex_normals[start_idx: end_idx: 1]


    def get_triangles_slice(self, slice_number=0):
        """
        Read split-triangles slice, to be used by WebGL visualizer.
        """
        if self.number_of_split_slices == 1:
            return self.triangles
        slice_number = int(slice_number)
        start_idx, end_idx = self._get_slice_triangle_boundaries(slice_number)
        return self.split_triangles[start_idx: end_idx: 1]


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
        for key, value in self._split_slices.iteritems():
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
        for triang in slice_triangles:
            # fixme: the library seems to assume here that all vertices are loaded
            # if we do that then why bother with the fancy lazy reads?
            # This is a mix of partial reads and full reads,
            # this needs both the surface datatype instance and the surfaceh5 class
            # fixme: these are reading from h5, the performance will be abysmal
            # maybe if the h5file manager would not close the file then assuming the h5 system caches small reads
            result_vertices.append(self.vertices[triang[0]])
            result_vertices.append(self.vertices[triang[1]])
            result_vertices.append(self.vertices[triang[2]])
        return numpy.array(result_vertices)

    def get_pick_vertex_normals_slice(self, slice_number=0):
        """
        Read vertex-normals slice, to be used by WebGL visualizer with pick.
        """
        slice_number = int(slice_number)
        slice_triangles = self.triangles[
            slice_number * SPLIT_PICK_MAX_TRIANGLE:
            min(self.number_of_triangles, (slice_number + 1) * SPLIT_PICK_MAX_TRIANGLE)
        ]
        result_normals = []
        for triang in slice_triangles:
            # fixme: these are reading from h5, the performance will be abysmal
            result_normals.append(self.vertex_normals[triang[0]])
            result_normals.append(self.vertex_normals[triang[1]])
            result_normals.append(self.vertex_normals[triang[2]])
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

    def generate_region_boundaries(self, region_mapping):
        """
        Return the full region boundaries, including: vertices, normals and lines indices.
        """
        boundary_vertices = []
        boundary_lines = []
        boundary_normals = []
        array_data = region_mapping.array_data

        for slice_idx in range(self._number_of_split_slices):
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

                lines_vert, lines_ind, lines_norm = Surface._process_triangle(triangle, reg_idx1, reg_idx2, dangling_idx,
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

# The h5 file format is not different for these surface subtypes
# so we should not create a new h5file
# If we will create a self.load -> Surface then that method should
# polymorphically decide on what surface subtype to construct
# class CorticalSurfaceH5(SurfaceH5):

