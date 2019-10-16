import json

from sqlalchemy import Column, Integer, ForeignKey, String, Float, Boolean
from sqlalchemy.orm import relationship
from tvb.datatypes.time_series import TimeSeriesRegion, TimeSeriesSurface, TimeSeriesVolume, TimeSeriesEEG, \
    TimeSeriesMEG, TimeSeriesSEEG

from tvb.core.entities.model.datatypes.sensors import SensorsIndex
from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.region_mapping import RegionMappingIndex, RegionVolumeMappingIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.entities.model.datatypes.volume import VolumeIndex
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neotraits.db import NArrayIndex



class TimeSeriesIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    data_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    data = relationship(NArrayIndex, foreign_keys=data_id, primaryjoin=NArrayIndex.id == data_id)

    time_id = Column(Integer, ForeignKey('narrays.id'))
    time = relationship(NArrayIndex, foreign_keys=time_id, primaryjoin=NArrayIndex.id == time_id)

    sample_period_unit = Column(String, nullable=False)
    sample_period = Column(Float, nullable=False)
    sample_rate = Column(Float)
    # length = Column(Float)
    labels_ordering = Column(String, nullable=False)
    has_volume_mapping = Column(Boolean, nullable=False, default=False)

    nr_dimensions = Column(Integer)
    length_1d = Column(Integer)
    length_2d = Column(Integer)
    length_3d = Column(Integer)
    length_4d = Column(Integer)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.title = datatype.title

        self.sample_period_unit = datatype.sample_period_unit
        self.sample_period = datatype.sample_period
        self.sample_rate = datatype.sample_rate
        self.labels_ordering = json.dumps(datatype.labels_ordering)

        # REVIEW THIS.
        # In general constructing graphs here is a bad ideea
        # But these NArrayIndex-es can be treated as part of this entity
        # never to be referenced by any other row or table.
        self.data = NArrayIndex.from_ndarray(datatype.data)
        self.time = NArrayIndex.from_ndarray(datatype.time)


class TimeSeriesEEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    sensors_id = Column(Integer, ForeignKey(SensorsIndex.id), nullable=not TimeSeriesEEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=sensors_id)


class TimeSeriesMEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    sensors_id = Column(Integer, ForeignKey(SensorsIndex.id), nullable=not TimeSeriesMEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=sensors_id)


class TimeSeriesSEEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    sensors_id = Column(Integer, ForeignKey(SensorsIndex.id), nullable=not TimeSeriesSEEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=sensors_id)


class TimeSeriesEEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    sensors_id = Column(Integer, ForeignKey(SensorsIndex.id), nullable=not TimeSeriesEEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=sensors_id)


class TimeSeriesMEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    sensors_id = Column(Integer, ForeignKey(SensorsIndex.id), nullable=not TimeSeriesMEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=sensors_id)


class TimeSeriesSEEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    sensors_id = Column(Integer, ForeignKey(SensorsIndex.id), nullable=not TimeSeriesSEEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=sensors_id)


class TimeSeriesRegionIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    connectivity_id = Column(Integer, ForeignKey(ConnectivityIndex.id),
                             nullable=not TimeSeriesRegion.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_id)

    region_mapping_volume_id = Column(Integer, ForeignKey(RegionVolumeMappingIndex.id),
                                      nullable=not TimeSeriesRegion.region_mapping_volume.required)
    region_mapping_volume = relationship(RegionVolumeMappingIndex, foreign_keys=region_mapping_volume_id)

    region_mapping_id = Column(Integer, ForeignKey(RegionMappingIndex.id),
                               nullable=not TimeSeriesRegion.region_mapping.required)
    region_mapping = relationship(RegionMappingIndex, foreign_keys=region_mapping_id)


class TimeSeriesSurfaceIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    surface_id = Column(Integer, ForeignKey(SurfaceIndex.id), nullable=not TimeSeriesSurface.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=surface_id)


class TimeSeriesVolumeIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    volume_id = Column(Integer, ForeignKey(VolumeIndex.id), nullable=not TimeSeriesVolume.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=volume_id)
