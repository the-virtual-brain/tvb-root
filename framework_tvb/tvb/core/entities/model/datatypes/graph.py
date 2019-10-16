from sqlalchemy import Column, Integer, ForeignKey, String, Float
from sqlalchemy.orm import relationship
from tvb.datatypes.graph import Covariance, CorrelationCoefficients, ConnectivityMeasure

from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neotraits.db import from_ndarray


class CovarianceIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not Covariance.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id, primaryjoin=TimeSeriesIndex.id == source_id)

    type = Column(String)

    def fill_from_has_traits(self, datatype):
        self.type = datatype.__class__.__name__
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.data_array)


class CorrelationCoefficientsIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    source_id = Column(Integer, ForeignKey(TimeSeriesIndex.id), nullable=not CorrelationCoefficients.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_id, primaryjoin=TimeSeriesIndex.id == source_id)

    type = Column(String)

    labels_ordering = Column(String)

    def fill_from_has_traits(self, datatype):
        self.type = datatype.__class__.__name__
        self.labels_ordering = datatype.labels_ordering
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.data_array)


class ConnectivityMeasureIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    type = Column(String)

    connectivity_id = Column(Integer, ForeignKey(ConnectivityIndex.id),
                             nullable=ConnectivityMeasure.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_id,
                                primaryjoin=ConnectivityIndex.id == connectivity_id)

    array_data_ndim = Column(Integer, nullable=False)
    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    def fill_from_has_traits(self, datatype):
        self.type = datatype.__class__.__name__
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.data_array)
