from sqlalchemy import Column, Integer, ForeignKey, String, Float, Boolean
from tvb.core.entities.model.model_datatype import DataType


class SurfaceIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    surface_type = Column(String, nullable=False)
    valid_for_simulations = Column(Boolean, nullable=False)
    number_of_vertices = Column(Integer, nullable=False)
    number_of_triangles = Column(Integer, nullable=False)
    number_of_edges = Column(Integer, nullable=False)
    bi_hemispheric = Column(Boolean, nullable=False)
    edge_length_mean = Column(Float, nullable=False)
    edge_length_min = Column(Float, nullable=False)
    edge_length_max = Column(Float, nullable=False)

    def fill_from_has_traits(self, datatype):
        self.surface_type = datatype.surface_type
        self.valid_for_simulations = datatype.valid_for_simulations
        self.number_of_vertices = datatype.number_of_vertices
        self.number_of_triangles = datatype.number_of_triangles
        self.number_of_edges = datatype.number_of_edges
        self.bi_hemispheric = datatype.bi_hemispheric
        self.edge_length_mean = datatype.edge_length_mean
        self.edge_length_min = datatype.edge_length_min
        self.edge_length_max = datatype.edge_length_max
