from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.mode_decompositions import PrincipalComponents

from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.entities.model.model_datatype import DataType


class PrincipalComponentsIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not PrincipalComponents.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id, primaryjoin=TimeSeriesIndex.id == source_id)

    type = Column(String)

    def fill_from_has_traits(self, datatype):
        self.type = datatype.__class__.__name__


class IndependentComponentsIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not PrincipalComponents.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id, primaryjoin=TimeSeriesIndex.id == source_id)

    type = Column(String)

    def fill_from_has_traits(self, datatype):
        self.type = datatype.__class__.__name__
