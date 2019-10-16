from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.graph import Covariance, CorrelationCoefficients, ConnectivityMeasure

from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.neotraits.db import HasTraitsIndex, NArrayIndex


class CovarianceIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_data = relationship(NArrayIndex, foreign_keys=array_data_id)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not Covariance.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id)

    type = Column(String)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.type = datatype.__class__.__name__
        self.array_data = NArrayIndex.from_ndarray(datatype.array_data)


class CorrelationCoefficientsIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_data = relationship(NArrayIndex, foreign_keys=array_data_id)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not CorrelationCoefficients.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id)

    type = Column(String)

    labels_ordering = Column(String)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.type = datatype.__class__.__name__
        self.labels_ordering = datatype.labels_ordering
        self.array_data = NArrayIndex.from_ndarray(datatype.array_data)


class ConnectivityMeasureIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    type = Column(String)

    connectivity_id = Column(Integer, ForeignKey(ConnectivityIndex.id),
                             nullable=ConnectivityMeasure.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_id)

    array_data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_data = relationship(NArrayIndex, foreign_keys=array_data_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.type = datatype.__class__.__name__
        self.array_data = NArrayIndex.from_ndarray(datatype.array_data)
