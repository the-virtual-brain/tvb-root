from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.patterns import StimuliRegion, StimuliSurface

from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.neotraits.db import HasTraitsIndex


class StimuliRegionIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    spatial_equation = Column(String, nullable=False)
    spatial_parameters = Column(String)
    temporal_equation = Column(String, nullable=False)
    temporal_parameters = Column(String)

    connectivity_id = Column(Integer, ForeignKey(ConnectivityIndex.id),
                             nullable=not StimuliRegion.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.spatial_equation = datatype.spatial.__class__.__name__
        self.spatial_parameters = datatype.spatial.parameters
        self.temporal_equation = datatype.temporal.__class__.__name__
        self.temporal_parameters = datatype.temporal.parameters


class StimuliSurfaceIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    spatial_equation = Column(String, nullable=False)
    spatial_parameters = Column(String)
    temporal_equation = Column(String, nullable=False)
    temporal_parameters = Column(String)

    surface_id = Column(Integer, ForeignKey(SurfaceIndex.id), nullable=not StimuliSurface.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=surface_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.spatial_equation = datatype.spatial.__class__.__name__
        self.spatial_parameters = datatype.spatial.parameters
        self.temporal_equation = datatype.temporal.__class__.__name__
        self.temporal_parameters = datatype.temporal.parameters
