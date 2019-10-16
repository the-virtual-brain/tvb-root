from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.mode_decompositions import PrincipalComponents

from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.neotraits.db import HasTraitsIndex


class PrincipalComponentsIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not PrincipalComponents.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id)

    type = Column(String)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.type = datatype.__class__.__name__


class IndependentComponentsIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not PrincipalComponents.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id)

    type = Column(String)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.type = datatype.__class__.__name__
