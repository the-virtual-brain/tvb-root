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
Backend-side for Visualizers that display measures on regions in the brain volume.

.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>
"""

import json
from abc import ABCMeta

from six import add_metaclass
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.basic.neotraits.api import Attr
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer, URLGenerator
from tvb.core.adapters.arguments_serialisation import *
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.model.model_datatype import DataTypeMatrix
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField, StrField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.structural import StructuralMRI


@add_metaclass(ABCMeta)
class _MappedArrayVolumeBase(ABCDisplayer):
    """
    Base functionality for all non-temporal volume views.
    It prepares for display a slice of a mapped array.
    """
    _ui_subsection = "volume"

    def get_required_memory_size(self, view_model):
        # type: (BaseVolumeVisualizerModel) -> int
        return -1

    @staticmethod
    def get_default_slice(measure_shape, nregions):
        default = [0 for _ in range(len(measure_shape))]
        for i in range(len(measure_shape)):
            if measure_shape[i] == nregions:
                default[i] = slice(None)
                return tuple(default)
        raise LaunchException("The mapped array of shape %s is incompatible with the region mapping "
                              "(expected values for %d connectivity regions)." % (measure_shape, nregions))

    def _ensure_region_mapping_index(self, region_mapping_volume, measure=None):
        # type: (RegionVolumeMappingIndex, DataTypeMatrix) -> RegionVolumeMappingIndex
        if region_mapping_volume is None:
            if measure is not None and hasattr(measure, "fk_connectivity_gid"):
                region_mapping_volume = dao.get_generic_entity(RegionVolumeMappingIndex,
                                                               measure.fk_connectivity_gid, 'fk_connectivity_gid')
                if region_mapping_volume is not None and len(region_mapping_volume):
                    region_mapping_volume = region_mapping_volume[0]
                else:
                    region_mapping_volume = None
            else:
                region_mapping_volume = dao.try_load_last_entity_of_type(self.current_project_id,
                                                                         RegionVolumeMappingIndex)
        if region_mapping_volume is None:
            raise LaunchException('You should have a compatible volume mapping to launch this viewer')
        return region_mapping_volume

    def _compute_region_volume_map_params(self, region_mapping_volume):
        # type: (RegionVolumeMappingIndex) -> dict
        # prepare the url that will display the region volume map
        conn_index = dao.get_datatype_by_gid(region_mapping_volume.fk_connectivity_gid)
        min_value, max_value = [0, conn_index.number_of_regions]
        url_volume_data = URLGenerator.build_url(self.stored_adapter.id, 'get_volume_view',
                                                 region_mapping_volume.gid, '')
        return dict(minValue=min_value, maxValue=max_value, urlVolumeData=url_volume_data)

    def _compute_measure_params(self, rvm_index, measure, data_slice):
        # type: (RegionVolumeMappingIndex, DataTypeMatrix, str) -> dict
        # prepare the url that will project the measure onto the region volume map
        measure_shape = measure.parsed_shape
        if not data_slice:
            conn_index = dao.get_datatype_by_gid(rvm_index.fk_connectivity_gid)
            data_slice = self.get_default_slice(measure_shape, conn_index.number_of_regions)
            data_slice = slice_str(data_slice)
        url_volume_data = URLGenerator.build_url(self.stored_adapter.id, 'get_mapped_array_volume_view',
                                                 rvm_index.gid, parameter='')
        url_volume_data += 'mapped_array_gid=' + measure.gid + ';mapped_array_slice=' + data_slice + ';'

        return dict(minValue=measure.array_data_min, maxValue=measure.array_data_max,
                    urlVolumeData=url_volume_data, measureShape=measure.shape, measureSlice=data_slice)

    def get_mapped_array_volume_view(self, entity_gid, mapped_array_gid, x_plane, y_plane, z_plane,
                                     mapped_array_slice=None, **kwargs):

        with h5.h5_file_for_gid(entity_gid) as entity_h5:
            data_shape = entity_h5.array_data.shape
            x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, data_shape[0],
                                                                    data_shape[1], data_shape[2])
            slice_x, slice_y, slice_z = entity_h5.get_volume_slice(x_plane, y_plane, z_plane)
            connectivity_gid = entity_h5.connectivity.load()

        with h5.h5_file_for_gid(mapped_array_gid) as mapped_array_h5:
            if mapped_array_slice:
                matrix_slice = parse_slice(mapped_array_slice)
                measure = mapped_array_h5.array_data[matrix_slice]
            else:
                measure = mapped_array_h5.array_data[:]

        connectivity_index = self.load_entity_by_gid(connectivity_gid)
        if measure.shape != (connectivity_index.number_of_regions,):
            raise ValueError('cannot project measure on the space')

        result_x = measure[slice_x]
        result_y = measure[slice_y]
        result_z = measure[slice_z]
        # Voxels outside the brain are -1. The indexing above is incorrect for those voxels as it
        # associates the values of the last region measure[-1] to them.
        # Here we replace those values with an out of scale value.
        result_x[slice_x == -1] = measure.min() - 1
        result_y[slice_y == -1] = measure.min() - 1
        result_z[slice_z == -1] = measure.min() - 1

        return [[result_x.tolist()],
                [result_y.tolist()],
                [result_z.tolist()]]

    @staticmethod
    def compute_background_params(min_value=0, max_value=0, url=None):
        return dict(minBackgroundValue=min_value, maxBackgroundValue=max_value, urlBackgroundVolumeData=url)

    def get_voxel_region(self, region_mapping_volume_gid, x_plane, y_plane, z_plane):
        with h5.h5_file_for_gid(region_mapping_volume_gid) as entity_h5:
            data_shape = entity_h5.array_data.shape
            x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, data_shape[0],
                                                                    data_shape[1], data_shape[2])
            slices = slice(x_plane, x_plane + 1), slice(y_plane, y_plane + 1), slice(z_plane, z_plane + 1)
            voxel = entity_h5.array_data[slices][0, 0, 0]
            connectivity_gid = entity_h5.connectivity.load()

        if voxel != -1:
            conn = self.load_traited_by_gid(connectivity_gid)
            return conn.region_labels[int(voxel)]
        else:
            return 'background'

    def compute_params(self, region_mapping_volume=None, measure=None, data_slice='', background=None):
        # type: (RegionVolumeMappingIndex, DataTypeMatrix, str, StructuralMRIIndex) -> dict

        region_mapping_volume = self._ensure_region_mapping_index(region_mapping_volume, measure)
        url_voxel_region = URLGenerator.build_url(self.stored_adapter.id, 'get_voxel_region', region_mapping_volume.gid,
                                                  parameter='')

        if measure is None:
            params = self._compute_region_volume_map_params(region_mapping_volume)
        else:
            params = self._compute_measure_params(region_mapping_volume, measure, data_slice)

        volume_gid = region_mapping_volume.fk_volume_gid
        volume_index = self.load_entity_by_gid(volume_gid)
        assert isinstance(volume_index, VolumeIndex)
        volume_shape = region_mapping_volume.parsed_shape
        volume_shape = (1,) + volume_shape

        params.update(volumeShape=json.dumps(volume_shape),
                      volumeOrigin=volume_index.origin,
                      voxelUnit=volume_index.voxel_unit,
                      voxelSize=volume_index.voxel_size,
                      urlVoxelRegion=url_voxel_region)

        if background is None:
            background = dao.try_load_last_entity_of_type(self.current_project_id, StructuralMRIIndex)
        if background is None:  # still
            params.update(self.compute_background_params())
        else:
            url_volume_data = URLGenerator.build_url(self.stored_adapter.id, 'get_volume_view', background.gid, '')
            params.update(self.compute_background_params(background.array_data_min,
                                                         background.array_data_max, url_volume_data))
        return params

    def get_volume_view(self, entity_gid, **kwargs):
        with h5.h5_file_for_gid(entity_gid) as ts_region_h5:
            if isinstance(ts_region_h5, TimeSeriesRegionH5):
                return self.prepare_view_region(ts_region_h5, **kwargs)

            volume_view = ts_region_h5.get_volume_view(**kwargs)
        return volume_view

    def prepare_view_region(self, ts_h5, x_plane, y_plane, z_plane, from_idx=None, to_idx=None, var=0, mode=0):
        region_mapping_volume_gid = ts_h5.region_mapping_volume.load()

        if region_mapping_volume_gid is None:
            raise Exception("Invalid method called for TS without Volume Mapping!")

        with h5.h5_file_for_gid(region_mapping_volume_gid) as volume_rm_h5:
            volume_rm_shape = volume_rm_h5.array_data.shape

            # Work with space inside Volume:
            x_plane, y_plane, z_plane = preprocess_space_parameters(x_plane, y_plane, z_plane, volume_rm_shape[0],
                                                                    volume_rm_shape[1], volume_rm_shape[2])
            region = self.get_view_region(ts_h5, volume_rm_h5, from_idx, to_idx, x_plane, y_plane, z_plane, var, mode)

        return region

    def get_view_region(self, ts_h5, volume_rm_h5, from_idx, to_idx, x_plane, y_plane, z_plane, var=0, mode=0):
        """
        Retrieve 3 slices through the Volume TS, at the given X, y and Z coordinates, and in time [from_idx .. to_idx].

        :param ts_h5: input TimeSeriesH5
        :param volume_rm_h5: input RegionVolumeMappingH5
        :param from_idx: int This will be the limit on the first dimension (time)
        :param to_idx: int Also limit on the first Dimension (time)
        :param x_plane: int coordinate
        :param y_plane: int coordinate
        :param z_plane: int coordinate

        :return: An array of 3 Matrices 2D, each containing the values to display in planes xy, yz and xy.
        """
        var, mode = int(var), int(mode)
        slice_x, slice_y, slice_z = volume_rm_h5.get_volume_slice(x_plane, y_plane, z_plane)

        # Read from the current TS:
        from_idx, to_idx, current_time_length = preprocess_time_parameters(from_idx, to_idx, ts_h5.data.shape[0])
        no_of_regions = ts_h5.data.shape[2]
        time_slices = slice(from_idx, to_idx), slice(var, var + 1), slice(no_of_regions), slice(mode, mode + 1)

        min_signal = ts_h5.get_min_max_values()[0]
        regions_ts = ts_h5.read_data_slice(time_slices)[:, 0, :, 0]
        regions_ts = numpy.hstack((regions_ts, numpy.ones((current_time_length, 1)) * ts_h5.out_of_range(min_signal)))

        # Index from TS with the space mapping:
        result_x, result_y, result_z = [], [], []

        for i in range(0, current_time_length):
            result_x.append(regions_ts[i][slice_x].tolist())
            result_y.append(regions_ts[i][slice_y].tolist())
            result_z.append(regions_ts[i][slice_z].tolist())

        return [result_x, result_y, result_z]


class BaseVolumeVisualizerModel(ViewModel):
    background = DataTypeGidAttr(
        linked_datatype=StructuralMRI,
        required=False,
        label='Background T1'
    )


@add_metaclass(ABCMeta)
class BaseVolumeVisualizerForm(ABCAdapterForm):

    def __init__(self):
        super(BaseVolumeVisualizerForm, self).__init__()
        self.background = TraitDataTypeSelectField(BaseVolumeVisualizerModel.background, name='background')


class VolumeVisualizerModel(BaseVolumeVisualizerModel):
    measure = DataTypeGidAttr(
        linked_datatype=DataTypeMatrix,
        label='Measure',
        doc='A measure to view on anatomy'
    )

    region_mapping_volume = DataTypeGidAttr(
        linked_datatype=RegionVolumeMapping,
        required=False,
        label='Region mapping'
    )

    data_slice = Attr(
        field_type=str,
        required=False,
        label='slice indices in numpy syntax'
    )


class VolumeVisualizerForm(BaseVolumeVisualizerForm):

    def __init__(self):
        super(VolumeVisualizerForm, self).__init__()
        self.measure = TraitDataTypeSelectField(VolumeVisualizerModel.measure, name='measure',
                                                conditions=self.get_filters())
        self.region_mapping_volume = TraitDataTypeSelectField(VolumeVisualizerModel.region_mapping_volume,
                                                              name='region_mapping_volume')
        self.data_slice = StrField(VolumeVisualizerModel.data_slice, name='data_slice')

    @staticmethod
    def get_view_model():
        return VolumeVisualizerModel

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.ndim',
                                   FilterChain.datatype + '.has_volume_mapping',
                                   FilterChain.datatype + '.subtype'],
                           operations=[">=", "==", "not in"],
                           values=[2, True, ["RegionVolumeMapping", "StructuralMRI"]])

    @staticmethod
    def get_input_name():
        return 'measure'

    @staticmethod
    def get_required_datatype():
        return DataTypeMatrix


class MappedArrayVolumeVisualizer(_MappedArrayVolumeBase):
    """
    This is a generic mapped array visualizer on a region volume.
    To view a multidimensional array one has to give this viewer a slice.
    """
    _ui_name = "Array Volume Visualizer"

    def get_form_class(self):
        return VolumeVisualizerForm

    def launch(self, view_model):
        # type: (VolumeVisualizerModel) -> dict
        measure_index = self.load_entity_by_gid(view_model.measure)
        region_mapping_volume_index = None
        background_volume_index = None

        if view_model.region_mapping_volume:
            region_mapping_volume_index = self.load_entity_by_gid(view_model.region_mapping_volume)
        if view_model.background:
            background_volume_index = self.load_entity_by_gid(view_model.background)

        params = self.compute_params(region_mapping_volume_index, measure_index, view_model.data_slice,
                                     background=background_volume_index)
        params['title'] = "Mapped array on region volume Visualizer"
        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))


class ConnectivityMeasureVolumeVisualizerModel(BaseVolumeVisualizerModel):
    connectivity_measure = DataTypeGidAttr(
        linked_datatype=ConnectivityMeasure,
        label='Connectivity measure',
        doc='A connectivity measure'
    )

    region_mapping_volume = DataTypeGidAttr(
        linked_datatype=RegionVolumeMapping,
        required=False,
        label='Region mapping'
    )


class ConnectivityMeasureVolumeVisualizerForm(BaseVolumeVisualizerForm):

    def __init__(self):
        super(ConnectivityMeasureVolumeVisualizerForm, self).__init__()
        self.connectivity_measure = TraitDataTypeSelectField(
            ConnectivityMeasureVolumeVisualizerModel.connectivity_measure, name='connectivity_measure',
            conditions=self.get_filters())
        self.region_mapping_volume = TraitDataTypeSelectField(
            ConnectivityMeasureVolumeVisualizerModel.region_mapping_volume, name='region_mapping_volume')

    @staticmethod
    def get_view_model():
        return ConnectivityMeasureVolumeVisualizerModel

    @staticmethod
    def get_required_datatype():
        return ConnectivityMeasureIndex

    @staticmethod
    def get_input_name():
        return 'connectivity_measure'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.ndim', FilterChain.datatype + '.has_volume_mapping'],
                           operations=["==", "=="], values=[1, True])


class ConnectivityMeasureVolumeVisualizer(_MappedArrayVolumeBase):
    _ui_name = "Connectivity Measure Volume Visualizer"

    def get_form_class(self):
        return ConnectivityMeasureVolumeVisualizerForm

    def launch(self, view_model):
        # type: (ConnectivityMeasureVolumeVisualizerModel) -> dict
        connectivity_measure_index = self.load_entity_by_gid(view_model.connectivity_measure)
        region_mapping_volume_index = None
        background_volume_index = None

        if view_model.region_mapping_volume:
            region_mapping_volume_index = self.load_entity_by_gid(view_model.region_mapping_volume)
        if view_model.background:
            background_volume_index = self.load_entity_by_gid(view_model.background)

        params = self.compute_params(region_mapping_volume_index, connectivity_measure_index,
                                     background=background_volume_index)
        params['title'] = "Connectivity Measure in Volume Visualizer"
        # the view will display slicing information if this key is present.
        # compute_params works with generic mapped arrays and it will return slicing info
        del params['measureSlice']
        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))


class RegionVolumeMappingVisualiserModel(BaseVolumeVisualizerModel):
    region_mapping_volume = DataTypeGidAttr(
        linked_datatype=RegionVolumeMapping,
        label='Region mapping'
    )

    connectivity_measure = DataTypeGidAttr(
        linked_datatype=ConnectivityMeasure,
        required=False,
        label='Connectivity measure',
        doc='A connectivity measure'
    )


class RegionVolumeMappingVisualiserForm(BaseVolumeVisualizerForm):

    def __init__(self):
        super(RegionVolumeMappingVisualiserForm, self).__init__()
        self.region_mapping_volume = TraitDataTypeSelectField(RegionVolumeMappingVisualiserModel.region_mapping_volume,
                                                              name='region_mapping_volume',
                                                              conditions=self.get_filters())

        cm_conditions = FilterChain(
            fields=[FilterChain.datatype + '.ndim', FilterChain.datatype + '.has_volume_mapping'],
            operations=["==", "=="], values=[1, True])
        self.connectivity_measure = TraitDataTypeSelectField(RegionVolumeMappingVisualiserModel.connectivity_measure,
                                                             name='connectivity_measure', conditions=cm_conditions)

    @staticmethod
    def get_view_model():
        return RegionVolumeMappingVisualiserModel

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return 'region_mapping_volume'

    @staticmethod
    def get_required_datatype():
        return RegionVolumeMappingIndex


class RegionVolumeMappingVisualiser(_MappedArrayVolumeBase):
    _ui_name = "Region Volume Mapping Visualizer"

    def get_form_class(self):
        return RegionVolumeMappingVisualiserForm

    def launch(self, view_model):
        # type: (RegionVolumeMappingVisualiserModel) -> dict

        connectivity_measure_index = None
        region_mapping_volume_index = None
        background_volume_index = None

        if view_model.connectivity_measure:
            connectivity_measure_index = self.load_entity_by_gid(view_model.connectivity_measure)
        if view_model.region_mapping_volume:
            region_mapping_volume_index = self.load_entity_by_gid(view_model.region_mapping_volume)
        if view_model.background:
            background_volume_index = self.load_entity_by_gid(view_model.background)

        params = self.compute_params(region_mapping_volume_index, connectivity_measure_index,
                                     background=background_volume_index)
        params['title'] = "Volume to Regions Visualizer"
        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))


class MriVolumeVisualizerForm(BaseVolumeVisualizerForm):

    def __init__(self):
        super(MriVolumeVisualizerForm, self).__init__()
        self.background.required = True

    @staticmethod
    def get_view_model():
        return BaseVolumeVisualizerModel

    @staticmethod
    def get_required_datatype():
        return StructuralMRIIndex

    @staticmethod
    def get_input_name():
        return 'background'

    @staticmethod
    def get_filters():
        return None


class MriVolumeVisualizer(_MappedArrayVolumeBase):
    _ui_name = "MRI Volume Visualizer"
    _ui_subsection = "volume"

    def get_form_class(self):
        return MriVolumeVisualizerForm

    def get_required_memory_size(self, view_model):
        # type: (BaseVolumeVisualizerModel) -> int
        return -1

    def launch(self, view_model):
        # type: (BaseVolumeVisualizerModel) -> dict

        structural_mri = self.load_entity_by_gid(view_model.background)
        assert isinstance(structural_mri, StructuralMRIIndex)

        volume_shape = structural_mri.parsed_shape
        volume_shape = (1,) + volume_shape

        url_volume_data = URLGenerator.build_url(self.stored_adapter.id, 'get_volume_view', view_model.background, '')

        volume_gid = structural_mri.fk_volume_gid
        volume_index = self.load_entity_by_gid(volume_gid)
        assert isinstance(volume_index, VolumeIndex)

        params = dict(title="MRI Volume visualizer",
                      minValue=structural_mri.array_data_min, maxValue=structural_mri.array_data_max,
                      urlVolumeData=url_volume_data, volumeShape=json.dumps(volume_shape),
                      volumeOrigin=volume_index.origin, voxelUnit=volume_index.voxel_unit,
                      voxelSize=volume_index.voxel_size,
                      urlVoxelRegion='', urlBackgroundVolumeData='',
                      minBackgroundValue=structural_mri.array_data_min,
                      maxBackgroundValue=structural_mri.array_data_max)

        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))
