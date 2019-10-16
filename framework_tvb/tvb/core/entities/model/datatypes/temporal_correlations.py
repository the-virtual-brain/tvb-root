import json

from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.temporal_correlations import CrossCorrelation

from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.neotraits.db import HasTraitsIndex, NArrayIndex


class CrossCorrelationIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_data = relationship(NArrayIndex, foreign_keys=array_data_id)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not CrossCorrelation.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id)

    labels_ordering = Column(String, nullable=False)
    type = Column(String)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_data = NArrayIndex.from_ndarray(datatype.array_data)
        self.labels_ordering = json.dumps(datatype.labels_ordering)
        self.type = datatype.__class__.__name__
