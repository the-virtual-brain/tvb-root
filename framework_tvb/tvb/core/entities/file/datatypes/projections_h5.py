from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Reference, Json

from tvb.datatypes.projections import ProjectionMatrix


class ProjectionMatrixH5(H5File):

    def __init__(self, path):
        super(ProjectionMatrixH5, self).__init__(path)
        self.projection_type = Scalar(ProjectionMatrix.projection_type, self)
        self.brain_skull = Reference(ProjectionMatrix.brain_skull, self)
        self.skull_skin = Reference(ProjectionMatrix.skull_skin, self)
        self.skin_air = Reference(ProjectionMatrix.skin_air, self)
        self.conductances = Json(ProjectionMatrix.conductances, self)
        self.sources = Reference(ProjectionMatrix.sources, self)
        self.sensors = Reference(ProjectionMatrix.sensors, self)
        self.projection_data = DataSet(ProjectionMatrix.projection_data, self)
