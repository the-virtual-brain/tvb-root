# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Backend-side for Visualizers that display measures on regions in the brain volume.

.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>
"""

import json
from tvb.basic.filters.chain import FilterChain
from tvb.basic.arguments_serialisation import slice_str
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import dao
from tvb.datatypes.arrays import MappedArray
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.structural import StructuralMRI


class _MappedArrayVolumeBase(ABCDisplayer):
    """
    Base functionality for all non-temporal volume views.
    It prepares for display a slice of a mapped array.
    """
    _ui_subsection = "volume"

    def get_required_memory_size(self, **kwargs):
        return -1

    @staticmethod
    def get_background_input_tree():
        return {'name': 'background', 'label': 'Background T1', 'type': StructuralMRI, 'required': False}

    @staticmethod
    def get_default_slice(measure_shape, nregions):
        default = [0 for _ in range(len(measure_shape))]
        for i in range(len(measure_shape)):
            if measure_shape[i] == nregions:
                default[i] = slice(None)
                return tuple(default)
        raise LaunchException("The mapped array of shape %s is incompatible with the region mapping "
                              "(expected values for %d connectivity regions)." %(measure_shape, nregions))


    def _ensure_region_mapping(self, region_mapping_volume):
        if region_mapping_volume is None:
            region_mapping_volume = dao.try_load_last_entity_of_type(self.current_project_id, RegionVolumeMapping)
        if region_mapping_volume is None:
            raise LaunchException('You should have a volume mapping to launch this viewer')
        return region_mapping_volume


    def _compute_region_volume_map_params(self, region_mapping_volume):
        # prepare the url that will display the region volume map
        min_value, max_value = [0, region_mapping_volume.connectivity.number_of_regions]
        url_volume_data = self.paths2url(region_mapping_volume, "get_volume_view", parameter="")
        return dict(minValue=min_value, maxValue=max_value,
                    urlVolumeData=url_volume_data)


    def _compute_measure_params(self, region_mapping_volume, measure, data_slice):
        # prepare the url that will project the measure onto the region volume map
        metadata = measure.get_metadata('array_data')
        min_value, max_value = metadata[measure.METADATA_ARRAY_MIN], metadata[measure.METADATA_ARRAY_MAX]
        measure_shape = measure.get_data_shape('array_data')
        if not data_slice:
            data_slice = self.get_default_slice(measure_shape, region_mapping_volume.connectivity.number_of_regions)
            data_slice = slice_str(data_slice)
        datatype_kwargs = json.dumps({'mapped_array': measure.gid})
        url_volume_data = ABCDisplayer.paths2url(region_mapping_volume, "get_mapped_array_volume_view")
        url_volume_data += '/' + datatype_kwargs + '?mapped_array_slice=' + data_slice + ';'

        return dict(minValue=min_value, maxValue=max_value,
                    urlVolumeData=url_volume_data,
                    measureShape=slice_str(measure_shape),
                    measureSlice=data_slice)

    @staticmethod
    def _compute_background(background):
        if background is not None:
            min_value, max_value = background.get_min_max_values()
            url_volume_data = ABCDisplayer.paths2url(background, 'get_volume_view', parameter='')
        else:
            min_value, max_value = 0, 0
            url_volume_data = None
        return dict(minBackgroundValue=min_value, maxBackgroundValue=max_value,
                    urlBackgroundVolumeData=url_volume_data)


    def compute_params(self, region_mapping_volume=None, measure=None, data_slice='', background=None):

        region_mapping_volume = self._ensure_region_mapping(region_mapping_volume)

        volume = region_mapping_volume.volume
        volume_shape = region_mapping_volume.read_data_shape()
        volume_shape = (1,) + volume_shape

        if measure is None:
            params = self._compute_region_volume_map_params(region_mapping_volume)
        else:
            params = self._compute_measure_params(region_mapping_volume, measure, data_slice)

        url_voxel_region = ABCDisplayer.paths2url(region_mapping_volume, "get_voxel_region", parameter="")

        params.update(volumeShape=json.dumps(volume_shape),
                      volumeOrigin=json.dumps(volume.origin.tolist()),
                      voxelUnit=volume.voxel_unit,
                      voxelSize=json.dumps(volume.voxel_size.tolist()),
                      urlVoxelRegion=url_voxel_region)

        if background is None:
            background = dao.try_load_last_entity_of_type(self.current_project_id, StructuralMRI)

        params.update(self._compute_background(background))
        return params


class MappedArrayVolumeVisualizer(_MappedArrayVolumeBase):
    """
    This is a generic mapped array visualizer on a region volume.
    To view a multidimensional array one has to give this viewer a slice.
    """
    _ui_name = "Array Volume Visualizer"


    def get_input_tree(self):
        return [{'name': 'measure', 'label': 'Measure',
                 'type': MappedArray, 'required': True,
                 'description': 'A measure to view on anatomy',
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=[">="], values=[2])},
                {'name': 'region_mapping_volume', 'label': 'Region mapping',
                 'type': RegionVolumeMapping, 'required': False, },
                {'name': 'data_slice', 'label': 'slice indices in numpy syntax',
                 'type': 'str', 'required': False},
                _MappedArrayVolumeBase.get_background_input_tree()]


    def launch(self, measure, region_mapping_volume=None, data_slice='', background=None):
        params = self.compute_params(region_mapping_volume, measure, data_slice, background=background)
        params['title'] = "Mapped array on region volume Visualizer",
        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))


class ConnectivityMeasureVolumeVisualizer(_MappedArrayVolumeBase):
    _ui_name = "Connectivity Measure Volume Visualizer"


    def get_input_tree(self):
        return [{'name': 'connectivity_measure', 'label': 'Connectivity measure',
                 'type': ConnectivityMeasure, 'required': True,
                 'description': 'A connectivity measure',
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1])},
                {'name': 'region_mapping_volume', 'label': 'Region mapping',
                 'type': RegionVolumeMapping, 'required': False, },
                _MappedArrayVolumeBase.get_background_input_tree()]


    def launch(self, connectivity_measure, region_mapping_volume=None, background=None):
        params = self.compute_params(region_mapping_volume, connectivity_measure, background=background)
        params['title'] = "Connectivity Measure in Volume Visualizer"
        # the view will display slicing information if this key is present.
        # compute_params works with generic mapped arrays and it will return slicing info
        del params['measureSlice']
        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))


class RegionVolumeMappingVisualiser(_MappedArrayVolumeBase):
    _ui_name = "Region Volume Mapping Visualizer"


    def get_input_tree(self):
        return [{'name': 'region_mapping_volume', 'label': 'Region mapping',
                 'type': RegionVolumeMapping, 'required': True, },
                {'name': 'connectivity_measure', 'label': 'Connectivity measure',
                 'type': ConnectivityMeasure, 'required': False,
                 'description': 'A connectivity measure',
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1])},
                _MappedArrayVolumeBase.get_background_input_tree()]


    def launch(self, region_mapping_volume, connectivity_measure=None, background=None):
        params = self.compute_params(region_mapping_volume, connectivity_measure, background=background)
        params['title'] = "Volume to Regions Visualizer"
        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))


class MriVolumeVisualizer(ABCDisplayer):

    _ui_name = "MRI Volume Visualizer"
    _ui_subsection = "volume"


    def get_required_memory_size(self, **kwargs):
        return -1


    def get_input_tree(self):
        tree = _MappedArrayVolumeBase.get_background_input_tree()
        tree['required'] = True
        return [tree]


    def launch(self, background=None):
        volume = background.volume
        volume_shape = background.read_data_shape()
        volume_shape = (1,) + volume_shape

        min_value, max_value = background.get_min_max_values()
        url_volume_data = ABCDisplayer.paths2url(background, 'get_volume_view', parameter='')

        params = dict(title="MRI Volume visualizer",
                      minValue=min_value, maxValue=max_value,
                      urlVolumeData=url_volume_data,
                      volumeShape=json.dumps(volume_shape),
                      volumeOrigin=json.dumps(volume.origin.tolist()),
                      voxelUnit=volume.voxel_unit,
                      voxelSize=json.dumps(volume.voxel_size.tolist()),
                      urlVoxelRegion='',
                      minBackgroundValue=min_value, maxBackgroundValue=max_value,
                      urlBackgroundVolumeData='')

        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))
