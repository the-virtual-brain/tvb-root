from tvb.basic.neotraits.api import Attr
from tvb.datatypes.patterns import StimuliRegion, StimuliSurface

from tvb.core.neotraits.h5 import H5File, Reference, DataSet, Scalar


class StimuliRegionH5(H5File):

    def __init__(self, path):
        super(StimuliRegionH5, self).__init__(path)
        self.spatial = Scalar(Attr(str), self, name='spatial')
        self.temporal = Scalar(Attr(str), self, name='temporal')
        self.connectivity = Reference(StimuliRegion.connectivity, self)
        self.weight = DataSet(StimuliRegion.weight, self)

    def store(self, datatype, scalars_only=False):
        self.connectivity.store(datatype.connectivity)
        self.weight.store(datatype.weight)
        self.spatial.store(datatype.spatial.to_json(datatype.spatial))
        self.temporal.store(datatype.temporal.to_json(datatype.temporal))

    def load_into(self, datatype):
        datatype.connectivity = self.connectivity.load()
        datatype.weight = self.weight.load()
        spatial_eq = self.spatial.load()
        spatial_eq = datatype.spatial.from_json(spatial_eq)
        datatype.spatial = spatial_eq
        temporal_eq = self.temporal.load()
        temporal_eq = datatype.temporal.from_json(temporal_eq)
        datatype.temporal = temporal_eq


class StimuliSurfaceH5(H5File):

    def __init__(self, path):
        super(StimuliSurfaceH5, self).__init__(path)
        self.spatial = Scalar(Attr(str), self, name='spatial')
        self.temporal = Scalar(Attr(str), self, name='temporal')
        self.surface = Reference(StimuliSurface.surface, self)
        self.focal_points_surface = DataSet(StimuliSurface.focal_points_surface, self)
        self.focal_points_triangles = DataSet(StimuliSurface.focal_points_triangles, self)

    def store(self, datatype, scalars_only=False):
        self.surface.store(datatype.surface)
        self.focal_points_surface.store(datatype.focal_points_surface)
        self.focal_points_triangles.store(datatype.focal_points_triangles)
        self.spatial.store(datatype.spatial.to_json(datatype.spatial))
        self.temporal.store(datatype.temporal.to_json(datatype.temporal))

    def load_into(self, datatype):
        datatype.surface = self.surface.load()
        datatype.focal_points_triangles = self.focal_points_triangles.load()
        datatype.focal_points_surface = self.focal_points_surface.load()
        spatial_eq = self.spatial.load()
        spatial_eq = datatype.spatial.from_json(spatial_eq)
        datatype.spatial = spatial_eq
        temporal_eq = self.temporal.load()
        temporal_eq = datatype.temporal.from_json(temporal_eq)
        datatype.temporal = temporal_eq
