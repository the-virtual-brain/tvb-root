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

"""
Backend-side for TS Visualizer of TS Volume DataTypes.

.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Robert Parcus <betoparcus@gmail.com>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ciprian Tomoiaga <ciprian.tomoiaga@codemart.ro>

"""

import json

import numpy
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesVolumeH5, TimeSeriesRegionH5
from tvb.adapters.visualizers.region_volume_mapping import _MappedArrayVolumeBase, MappedArrayVolumeVisualizer
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.adapters.arguments_serialisation import postprocess_voxel_ts
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.utils import prepare_time_slice
from tvb.datatypes.structural import StructuralMRI
from tvb.datatypes.time_series import TimeSeries


class TimeSeriesVolumeVisualiserModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label='Time Series'
    )

    background = DataTypeGidAttr(
        linked_datatype=StructuralMRI,
        required=False,
        label='Background T1'
    )


class TimeSeriesVolumeVisualiserForm(ABCAdapterForm):

    def __init__(self):
        super(TimeSeriesVolumeVisualiserForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(TimeSeriesVolumeVisualiserModel.time_series, name='time_series',
                                                    conditions=self.get_filters())
        self.background = TraitDataTypeSelectField(TimeSeriesVolumeVisualiserModel.background, name='background')

    @staticmethod
    def get_view_model():
        return TimeSeriesVolumeVisualiserModel

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.has_volume_mapping'], operations=["=="], values=[True])

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex


class TimeSeriesVolumeVisualiser(_MappedArrayVolumeBase):
    _ui_name = "Time Series Volume Visualizer"
    _ui_subsection = "volume"

    def get_form_class(self):
        return TimeSeriesVolumeVisualiserForm

    def get_required_memory_size(self, view_model):
        # type: (TimeSeriesVolumeVisualiserModel) -> int
        """Return required memory."""
        return -1

    def launch(self, view_model):
        # type: (TimeSeriesVolumeVisualiserModel) -> dict

        url_volume_data = URLGenerator.build_url(MappedArrayVolumeVisualizer.stored_adapter.id, 'get_volume_view', view_model.time_series, '')
        url_timeseries_data = URLGenerator.build_url(self.stored_adapter.id, 'get_voxel_time_series',
                                                     view_model.time_series, '')

        ts_index = self.load_entity_by_gid(view_model.time_series)
        ts_h5 = h5.h5_file_for_index(ts_index)
        min_value, max_value = ts_h5.get_min_max_values()

        if isinstance(ts_h5, TimeSeriesVolumeH5):
            volume_shape = ts_h5.data.shape
            volume = self.load_traited_by_gid(ts_h5.volume.load())
        else:
            rm_index = self.load_entity_by_gid(ts_h5.region_mapping_volume.load())
            volume = self.load_traited_by_gid(rm_index.fk_volume_gid)
            rmv = self.load_traited_by_gid(ts_h5.region_mapping_volume.load())
            volume_shape = [ts_h5.data.shape[0]]
            volume_shape.extend(rmv.array_data.shape)

        background_index = None
        if view_model.background:
            background_index = self.load_entity_by_gid(view_model.background)

        params = dict(title="Volumetric Time Series",
                      ts_title=ts_h5.title.load(),
                      labelsStateVar=ts_index.get_labels_for_dimension(1),
                      labelsModes=list(range(ts_index.data_length_4d)),
                      minValue=min_value, maxValue=max_value,
                      urlVolumeData=url_volume_data,
                      urlTimeSeriesData=url_timeseries_data,
                      samplePeriod=ts_h5.sample_period.load(),
                      samplePeriodUnit=ts_h5.sample_period_unit.load(),
                      volumeShape=json.dumps(volume_shape),
                      volumeOrigin=json.dumps(volume.origin.tolist()),
                      voxelUnit=volume.voxel_unit,
                      voxelSize=json.dumps(volume.voxel_size.tolist()))

        params.update(self.ensure_background(background_index))
        ts_h5.close()

        return self.build_display_result("time_series_volume/view", params,
                                         pages=dict(controlPage="time_series_volume/controls"))

    def ensure_background(self, background_index):
        if background_index is None:
            background_index = dao.try_load_last_entity_of_type(self.current_project_id, StructuralMRIIndex)

        if background_index is None:
            return _MappedArrayVolumeBase.compute_background_params()

        with h5.h5_file_for_index(background_index) as background_h5:
            min_value, max_value = background_h5.get_min_max_values()

        url_volume_data = URLGenerator.build_url(self.stored_adapter.id, 'get_volume_view', background_index.gid, '')
        return _MappedArrayVolumeBase.compute_background_params(min_value, max_value, url_volume_data)

    def get_voxel_time_series(self, entity_gid, **kwargs):
        """
        Retrieve for a given voxel (x,y,z) the entire timeline.

        :param x: int coordinate
        :param y: int coordinate
        :param z: int coordinate

        :return: A complex dictionary with information about current voxel.
                The main part will be a vector with all the values over time from the x,y,z coordinates.
        """

        with h5.h5_file_for_gid(entity_gid) as ts_h5:
            if isinstance(ts_h5, TimeSeriesRegionH5):
                return self.prepare_view_region(ts_h5, **kwargs)

            data = ts_h5.get_voxel_time_series(**kwargs)
        return data

    def get_view_region(self, ts_h5, volume_rm_h5, from_idx, to_idx, x, y, z, var=0, mode=0):
        idx_slices = slice(x, x + 1), slice(y, y + 1), slice(z, z + 1)

        idx = int(volume_rm_h5.array_data[idx_slices])

        time_length = ts_h5.data.shape[0]
        var, mode = int(var), int(mode)
        voxel_slices = prepare_time_slice(time_length), slice(var, var + 1), slice(idx, idx + 1), slice(mode, mode + 1)

        connectivity_gid = volume_rm_h5.connectivity.load()
        with h5.h5_file_for_gid(connectivity_gid) as connectivity_h5:
            label = connectivity_h5.region_labels.load()[idx]

        background, back_min, back_max = None, None, None
        if idx < 0:
            back_min, back_max = ts_h5.get_min_max_values()
            background = numpy.ones((time_length, 1)) * ts_h5.out_of_range(back_min)
            label = 'background'

        result = postprocess_voxel_ts(ts_h5, voxel_slices, background, back_min, back_max, label)
        return result
