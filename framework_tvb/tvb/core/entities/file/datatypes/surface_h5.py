from tvb.core.neotraits.h5 import H5File, DataSet, Scalar
from tvb.datatypes.surfaces import Surface


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
        # self.split_slices = Scalar(Surface.split_slices, self)
        self.bi_hemispheric = Scalar(Surface.bi_hemispheric, self)
        self.surface_type = Scalar(Surface.surface_type, self)
        self.valid_for_simulations = Scalar(Surface.valid_for_simulations, self)


class CorticalSurfaceH5(SurfaceH5):

    def __init__(self, path):
        super(CorticalSurfaceH5, self).__init__(path)