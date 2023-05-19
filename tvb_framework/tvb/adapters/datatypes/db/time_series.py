# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#
import json
from tvb.datatypes.time_series import *
from sqlalchemy import Column, Integer, ForeignKey, String, Float, Boolean
from sqlalchemy.orm import relationship
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex, RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.model.model_datatype import DataType


class TimeSeriesIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    time_series_type = Column(String, nullable=False)
    data_ndim = Column(Integer, nullable=False)
    data_length_1d = Column(Integer)
    data_length_2d = Column(Integer)
    data_length_3d = Column(Integer)
    data_length_4d = Column(Integer)
    start_time = Column(Float, default=0)

    sample_period_unit = Column(String, nullable=False)
    sample_period = Column(Float, nullable=False)
    sample_rate = Column(Float)
    labels_ordering = Column(String, nullable=False)
    labels_dimensions = Column(String, nullable=False)
    has_volume_mapping = Column(Boolean, nullable=False, default=False)
    has_surface_mapping = Column(Boolean, nullable=False, default=False)

    def get_extra_info(self):
        labels_dict = {}
        labels_dict["labels_ordering"] = self.labels_ordering
        labels_dict["labels_dimensions"] = self.labels_dimensions
        return labels_dict

    def fill_from_has_traits(self, datatype):
        # type: (TimeSeries)  -> None
        super(TimeSeriesIndex, self).fill_from_has_traits(datatype)
        self.title = datatype.title
        self.time_series_type = type(datatype).__name__
        self.start_time = datatype.start_time
        self.sample_period_unit = datatype.sample_period_unit
        self.sample_period = datatype.sample_period
        self.sample_rate = datatype.sample_rate
        self.labels_ordering = json.dumps(datatype.labels_ordering)
        self.labels_dimensions = json.dumps(datatype.labels_dimensions)

        # REVIEW THIS.
        # In general constructing graphs here is a bad ideea
        # But these NArrayIndex-es can be treated as part of this entity
        # never to be referenced by any other row or table.
        if hasattr(datatype, 'data'):
            self.data_ndim = datatype.data.ndim
            self.fill_shape(datatype.data.shape)

    def fill_from_h5(self, h5_file):
        super(TimeSeriesIndex, self).fill_from_h5(h5_file)
        self.time_series_type = type(h5_file).__name__.replace('H5', '')
        self.title = h5_file.title.load()
        self.start_time = h5_file.start_time.load()
        self.sample_period_unit = h5_file.sample_period_unit.load()
        self.sample_period = h5_file.sample_period.load()
        self.sample_rate = h5_file.sample_rate.load()
        self.labels_ordering = json.dumps(h5_file.labels_ordering.load())
        self.labels_dimensions = json.dumps(h5_file.labels_dimensions.load())
        self.fill_shape(h5_file.data.shape)

    def fill_shape(self, final_shape):
        self.data_ndim = len(final_shape)
        self.data_length_1d = final_shape[0]
        if self.data_ndim > 1:
            self.data_length_2d = final_shape[1]
            if self.data_ndim > 2:
                self.data_length_3d = final_shape[2]
                if self.data_ndim > 3:
                    self.data_length_4d = final_shape[3]

    @staticmethod
    def accepted_filters():
        filters = DataType.accepted_filters()
        filters.update(
            {FilterChain.datatype + '.data_ndim':
                 {'type': 'int', 'display': 'No of Dimensions', 'operations': ['==', '<', '>']},
             FilterChain.datatype + '.sample_period':
                 {'type': 'float', 'display': 'Sample Period', 'operations': ['==', '<', '>']},
             FilterChain.datatype + '.sample_rate':
                 {'type': 'float', 'display': 'Sample Rate', 'operations': ['==', '<', '>']},
             FilterChain.datatype + '.title':
                 {'type': 'string', 'display': 'Title', 'operations': ['==', '!=', 'like']}
             })
        return filters

    def get_data_shape(self):
        if self.data_ndim == 1:
            return self.data_length_1d
        if self.data_ndim == 2:
            return self.data_length_1d, self.data_length_2d
        if self.data_ndim == 3:
            return self.data_length_1d, self.data_length_2d, self.data_length_3d
        return self.data_length_1d, self.data_length_2d, self.data_length_3d, self.data_length_4d

    def get_labels_for_dimension(self, idx):
        label_dimensions = json.loads(self.labels_dimensions)
        labels_ordering = json.loads(self.labels_ordering)
        return label_dimensions.get(labels_ordering[idx], ["0"])


class TimeSeriesEEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    fk_sensors_gid = Column(String(32), ForeignKey(SensorsIndex.gid), nullable=not TimeSeriesEEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=fk_sensors_gid)

    def fill_from_has_traits(self, datatype):
        # type: (TimeSeriesEEG)  -> None
        super(TimeSeriesEEGIndex, self).fill_from_has_traits(datatype)
        self.fk_sensors_gid = datatype.sensors.gid.hex
        # Because we had a ProjectionMatrix in the monitor
        self.has_surface_mapping = True

    def fill_from_h5(self, h5_file):
        super(TimeSeriesEEGIndex, self).fill_from_h5(h5_file)
        self.fk_sensors_gid = h5_file.sensors.load().hex
        self.has_surface_mapping = True


class TimeSeriesMEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    fk_sensors_gid = Column(String(32), ForeignKey(SensorsIndex.gid), nullable=not TimeSeriesMEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=fk_sensors_gid)

    def fill_from_has_traits(self, datatype):
        # type: (TimeSeriesMEG)  -> None
        super(TimeSeriesMEGIndex, self).fill_from_has_traits(datatype)
        self.fk_sensors_gid = datatype.sensors.gid.hex
        self.has_surface_mapping = True

    def fill_from_h5(self, h5_file):
        super(TimeSeriesMEGIndex, self).fill_from_h5(h5_file)
        self.fk_sensors_gid = h5_file.sensors.load().hex
        self.has_surface_mapping = True


class TimeSeriesSEEGIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    fk_sensors_gid = Column(String(32), ForeignKey(SensorsIndex.gid), nullable=not TimeSeriesSEEG.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=fk_sensors_gid)

    def fill_from_has_traits(self, datatype):
        # type: (TimeSeriesSEEG)  -> None
        super(TimeSeriesSEEGIndex, self).fill_from_has_traits(datatype)
        self.fk_sensors_gid = datatype.sensors.gid.hex
        self.has_surface_mapping = True

    def fill_from_h5(self, h5_file):
        super(TimeSeriesSEEGIndex, self).fill_from_h5(h5_file)
        self.fk_sensors_gid = h5_file.sensors.load().hex
        self.has_surface_mapping = True


class TimeSeriesRegionIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    fk_connectivity_gid = Column(String(32), ForeignKey(ConnectivityIndex.gid),
                                 nullable=not TimeSeriesRegion.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=fk_connectivity_gid,
                                primaryjoin=ConnectivityIndex.gid == fk_connectivity_gid)

    fk_region_mapping_volume_gid = Column(String(32), ForeignKey(RegionVolumeMappingIndex.gid),
                                          nullable=not TimeSeriesRegion.region_mapping_volume.required)
    region_mapping_volume = relationship(RegionVolumeMappingIndex, foreign_keys=fk_region_mapping_volume_gid,
                                         primaryjoin=RegionVolumeMappingIndex.gid == fk_region_mapping_volume_gid)

    fk_region_mapping_gid = Column(String(32), ForeignKey(RegionMappingIndex.gid),
                                   nullable=not TimeSeriesRegion.region_mapping.required)
    region_mapping = relationship(RegionMappingIndex, foreign_keys=fk_region_mapping_gid,
                                  primaryjoin=RegionMappingIndex.gid == fk_region_mapping_gid)

    def fill_from_has_traits(self, datatype):
        # type: (TimeSeriesRegion)  -> None
        super(TimeSeriesRegionIndex, self).fill_from_has_traits(datatype)
        self.fk_connectivity_gid = datatype.connectivity.gid.hex
        if datatype.region_mapping_volume is not None:
            self.fk_region_mapping_volume_gid = datatype.region_mapping_volume.gid.hex
            self.has_volume_mapping = True
        if datatype.region_mapping is not None:
            self.fk_region_mapping_gid = datatype.region_mapping.gid.hex
            self.has_surface_mapping = True

    def fill_from_h5(self, h5_file):
        super(TimeSeriesRegionIndex, self).fill_from_h5(h5_file)
        self.fk_connectivity_gid = h5_file.connectivity.load().hex
        region_mapping_volume = h5_file.region_mapping_volume.load()
        if region_mapping_volume is not None:
            self.fk_region_mapping_volume_gid = region_mapping_volume.hex
            self.has_volume_mapping = True
        region_mapping = h5_file.region_mapping.load()
        if region_mapping is not None:
            self.fk_region_mapping_gid = region_mapping.hex
            self.has_surface_mapping = True


class TimeSeriesSurfaceIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    fk_surface_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not TimeSeriesSurface.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=fk_surface_gid)

    def fill_from_has_traits(self, datatype):
        # type: (TimeSeriesSurface)  -> None
        super(TimeSeriesSurfaceIndex, self).fill_from_has_traits(datatype)
        self.fk_surface_gid = datatype.surface.gid.hex
        self.has_surface_mapping = True

    def fill_from_h5(self, h5_file):
        super(TimeSeriesSurfaceIndex, self).fill_from_h5(h5_file)
        self.fk_surface_gid = h5_file.surface.load().hex
        self.has_surface_mapping = True


class TimeSeriesVolumeIndex(TimeSeriesIndex):
    id = Column(Integer, ForeignKey(TimeSeriesIndex.id), primary_key=True)

    fk_volume_gid = Column(String(32), ForeignKey(VolumeIndex.gid), nullable=not TimeSeriesVolume.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=fk_volume_gid)

    def fill_from_has_traits(self, datatype):
        # type: (TimeSeriesVolume)  -> None
        super(TimeSeriesVolumeIndex, self).fill_from_has_traits(datatype)
        self.fk_volume_gid = datatype.volume.gid.hex
        self.has_volume_mapping = True

    def fill_from_h5(self, h5_file):
        super(TimeSeriesVolumeIndex, self).fill_from_h5(h5_file)
        self.fk_volume_gid = h5_file.volume.load().hex
        self.has_volume_mapping = True
