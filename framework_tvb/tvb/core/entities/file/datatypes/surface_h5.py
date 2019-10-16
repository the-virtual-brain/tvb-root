import logging
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Json
from tvb.datatypes.surfaces import Surface, KEY_VERTICES, KEY_START, KEY_END, SPLIT_BUFFER_SIZE

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
        self._headers_loaded = False
        self._number_of_vertices = None
        self._split_slices = None


    def _ensure_headers(self):
        """
        Load the data that is required in order to interpret the file format
        number_of_vertices and split_slices are needed for the get_vertices_slice read call
        """
        if self._headers_loaded:
            return
        self._number_of_vertices = self.number_of_vertices.load()
        self._split_slices = self.split_slices.load()


    # experimental port of some of the data access apis from the datatype


    def _get_slice_vertex_boundaries(self, slice_idx):
        self._ensure_headers()

        if str(slice_idx) in self._split_slices:
            start_idx = max(0, self._split_slices[str(slice_idx)][KEY_VERTICES][KEY_START])
            end_idx = min(self._split_slices[str(slice_idx)][KEY_VERTICES][KEY_END], self._number_of_vertices)
            return start_idx, end_idx
        else:
            log.warning("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, min(SPLIT_BUFFER_SIZE, self._number_of_vertices)


    def get_vertices_slice(self, slice_number=0):
        """
        Read vertices slice, to be used by WebGL visualizer.
        """
        slice_number = int(slice_number)
        start_idx, end_idx = self._get_slice_vertex_boundaries(slice_number)
        return self.vertices[start_idx: end_idx: 1]


# The h5 file format is not different for these surface subtypes
# so we should not create a new h5file
# If we will create a self.load -> Surface then that method should
# polymorphically decide on what surface subtype to construct
# class CorticalSurfaceH5(SurfaceH5):

