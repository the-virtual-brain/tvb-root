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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy

from tvb.adapters.datatypes.db.time_series import *
from tvb.adapters.datatypes.h5.surface_h5 import SurfaceH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5, TimeSeriesSurfaceH5
from tvb.adapters.visualizers.eeg_monitor import EegMonitor
from tvb.adapters.visualizers.sensors import prepare_mapped_sensors_as_measure_points_params
from tvb.adapters.visualizers.sensors import prepare_sensors_as_measure_points_params, function_sensors_to_surface
from tvb.adapters.visualizers.surface_view import ensure_shell_surface, SurfaceURLGenerator, ABCSurfaceDisplayer
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import DataTypeGidAttr, ViewModel
from tvb.datatypes.surfaces import Surface, SurfaceTypesEnum

MAX_MEASURE_POINTS_LENGTH = 600


class BrainViewerModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label='Time Series (Region or Surface)'
    )

    shell_surface = DataTypeGidAttr(
        linked_datatype=Surface,
        required=False,
        label='Shell Surface',
        doc='Surface to be displayed semi-transparently as overlay, for visual navigation purposes only.'
    )


class BrainViewerForm(ABCAdapterForm):

    def __init__(self):
        super(BrainViewerForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(BrainViewerModel.time_series, name='time_series',
                                                    conditions=self.get_filters())
        self.shell_surface = TraitDataTypeSelectField(BrainViewerModel.shell_surface, name='shell_surface')

    @staticmethod
    def get_view_model():
        return BrainViewerModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(
            fields=[FilterChain.datatype + '.time_series_type', FilterChain.datatype + '.has_surface_mapping'],
            operations=["in", "=="], values=[['TimeSeriesRegion', 'TimeSeriesSurface'], True])


class BrainViewer(ABCSurfaceDisplayer):
    """
    Interface between the 3D view of the Brain Cortical Surface and TVB framework.
    This viewer will build the required parameter dictionary that will be sent to the HTML / JS for further processing,
    having as end result a brain surface plus activity that will be displayed in 3D.
    """
    _ui_name = "Brain Activity Visualizer"
    PAGE_SIZE = 500

    def get_form_class(self):
        return BrainViewerForm

    def get_required_memory_size(self, view_model):
        # type: (BrainViewerModel) -> numpy.ndarray
        """
        Assume one page doesn't get 'dumped' in time and it is highly probably that
        two consecutive pages will be in the same time in memory.
        """
        time_series = self.load_entity_by_gid(view_model.time_series)
        overall_shape = time_series.get_data_shape()
        used_shape = (overall_shape[0] / (self.PAGE_SIZE * 2.0), overall_shape[1], overall_shape[2], overall_shape[3])
        return numpy.prod(used_shape) * 8.0

    def launch(self, view_model):
        # type: (BrainViewerModel) -> dict
        """
        Build visualizer's page.
        """
        time_series_index = self.load_entity_by_gid(view_model.time_series)
        shell_surface_index = None
        if view_model.shell_surface:
            shell_surface_index = self.load_entity_by_gid(view_model.shell_surface)
        params = self.compute_parameters(time_series_index, shell_surface_index)
        return self.build_display_result("brain/view", params, pages=dict(controlPage="brain/controls"))

    def populate_surface_fields(self, time_series_index):
        """
        To be overwritten for populating fields: one_to_one_map/connectivity/region_map/surface fields
        """

        self.one_to_one_map = isinstance(time_series_index, TimeSeriesSurfaceIndex)

        if self.one_to_one_map:
            self.PAGE_SIZE /= 10
            surface_gid = time_series_index.fk_surface_gid
            surface_index = dao.get_datatype_by_gid(surface_gid)
            region_map_indexes = dao.get_generic_entity(RegionMappingIndex, surface_gid, 'fk_surface_gid')
            if len(region_map_indexes) < 1:
                region_map_index = None
                connectivity_index = None
            else:
                region_map_index = region_map_indexes[0]
                connectivity_index = dao.get_datatype_by_gid(region_map_index.fk_connectivity_gid)
        else:
            connectivity_index = dao.get_datatype_by_gid(time_series_index.fk_connectivity_gid)

            if time_series_index.fk_region_mapping_gid:
                region_map_index = dao.get_datatype_by_gid(time_series_index.fk_region_mapping_gid)
            else:
                region_map_indexes = dao.get_generic_entity(RegionMappingIndex, connectivity_index.gid,
                                                            'fk_connectivity_gid')
                region_map_index = region_map_indexes[0]

            surface_index = dao.get_datatype_by_gid(region_map_index.fk_surface_gid)

        self.connectivity_index = connectivity_index
        self.region_map_gid = None if region_map_index is None else region_map_index.gid
        self.surface_gid = None if surface_index is None else surface_index.gid
        self.surface_h5 = None if surface_index is None else h5.h5_file_for_index(surface_index)

    def retrieve_measure_points_params(self, time_series):
        """
        To be overwritten method, for retrieving the measurement points (region centers, EEG sensors).
        """
        if self.connectivity_index is None:
            self.measure_points_no = 0
            return {'urlMeasurePoints': [],
                    'urlMeasurePointsLabels': [],
                    'noOfMeasurePoints': 0}

        connectivity_gid = self.connectivity_index.gid
        measure_points = SurfaceURLGenerator.build_h5_url(connectivity_gid, 'get_centres')
        measure_points_labels = SurfaceURLGenerator.build_h5_url(connectivity_gid, 'get_region_labels')
        self.measure_points_no = self.connectivity_index.number_of_regions

        return {'urlMeasurePoints': measure_points,
                'urlMeasurePointsLabels': measure_points_labels,
                'noOfMeasurePoints': self.measure_points_no}

    def compute_parameters(self, time_series, shell_surface=None):
        """
        Create the required parameter dictionary for the HTML/JS viewer.

        :rtype: `dict`
        :raises Exception: when
                    * number of measure points exceeds the maximum allowed
                    * a Face object cannot be found in database

        """
        self.populate_surface_fields(time_series)

        url_vertices, url_normals, url_lines, url_triangles, url_region_map = SurfaceURLGenerator.get_urls_for_rendering(
            self.surface_h5, self.region_map_gid)
        hemisphere_chunk_mask = self.surface_h5.get_slices_to_hemisphere_mask()

        params = self.retrieve_measure_points_params(time_series)

        if not self.one_to_one_map and self.measure_points_no > MAX_MEASURE_POINTS_LENGTH:
            raise Exception("Max number of measure points " + str(MAX_MEASURE_POINTS_LENGTH) + " exceeded.")

        time_series_h5 = h5.h5_file_for_index(time_series)
        assert isinstance(time_series_h5, TimeSeriesH5)
        base_adapter_url, time_urls = self._prepare_data_slices(time_series)
        min_val, max_val = time_series_h5.get_min_max_values()
        legend_labels = self._compute_legend_labels(min_val, max_val)

        state_variables = time_series.get_labels_for_dimension(1)

        if self.surface_gid and self.region_map_gid:
            boundary_url = SurfaceURLGenerator.get_url_for_region_boundaries(self.surface_gid, self.region_map_gid,
                                                                             self.stored_adapter.id)
        else:
            boundary_url = ''

        shell_surface = ensure_shell_surface(self.current_project_id, shell_surface)
        params.update(dict(title="Cerebral Activity: " + time_series.title, isOneToOneMapping=self.one_to_one_map,
                           urlVertices=json.dumps(url_vertices), urlTriangles=json.dumps(url_triangles),
                           urlLines=json.dumps(url_lines), urlNormals=json.dumps(url_normals),
                           urlRegionMap=json.dumps(url_region_map), base_adapter_url=base_adapter_url,
                           time=json.dumps(time_urls), minActivity=min_val, maxActivity=max_val,
                           legendLabels=legend_labels, labelsStateVar=state_variables,
                           labelsModes=list(range(time_series.data_length_4d)), extended_view=False,
                           shellObject=self.prepare_shell_surface_params(shell_surface, SurfaceURLGenerator),
                           biHemispheric=self.surface_h5.bi_hemispheric.load(),
                           hemisphereChunkMask=json.dumps(hemisphere_chunk_mask),
                           pageSize=self.PAGE_SIZE, urlRegionBoundaries=boundary_url,
                           measurePointsLabels=self.get_space_labels(time_series_h5),
                           measurePointsTitle=time_series.title))

        params.update(self.build_params_for_subselectable_ts(time_series_h5))

        time_series_h5.close()
        if self.surface_h5:
            self.surface_h5.close()

        return params

    @staticmethod
    def _prepare_mappings(mappings_dict):
        """
        Get full mapping dictionary between the original vertices and multiple slices (for WebGL compatibility).
        """
        prepared_mappings = []
        for key in mappings_dict:
            this_mappings = []
            vert_map_dict = mappings_dict[key]
            vertices_indexes = vert_map_dict['indices']
            this_mappings.append(vertices_indexes[0].tolist())
            for i in range(1, len(vertices_indexes)):
                if vertices_indexes[i][0] == vertices_indexes[i][1]:
                    this_mappings.append(vertices_indexes[i][0])
                else:
                    for index in range(vertices_indexes[i][0], vertices_indexes[i][1] + 1):
                        this_mappings.append(index)
            prepared_mappings.append(this_mappings)
        return prepared_mappings

    @staticmethod
    def _compute_legend_labels(min_val, max_val, nr_labels=5, min_nr_dec=3):
        """
        Compute rounded labels for MIN and MAX values such that decimals will show a difference between them.
        """
        if len(str(min_val).split('.')) == 2:
            min_integer, min_decimals = str(min_val).split('.')
        else:
            min_integer, min_decimals = [str(int(min_val)), ""]
        if len(str(max_val).split('.')) == 2:
            max_integer, max_decimals = str(max_val).split('.')
        else:
            max_integer, max_decimals = [str(int(max_val)), ""]
        idx = min_nr_dec
        if len(min_decimals) < min_nr_dec or len(max_decimals) < min_nr_dec:
            processed_min_val = float(min_val)
            processed_max_val = float(max_val)
        elif min_integer != max_integer:
            processed_min_val = float(min_integer + '.' + min_decimals[:min_nr_dec])
            processed_max_val = float(max_integer + '.' + max_decimals[:min_nr_dec])
        else:
            for idx, val in enumerate(min_decimals):
                if idx < len(max_decimals) or val != max_decimals[idx]:
                    break
            processed_min_val = float(min_integer + '.' + min_decimals[:idx])
            processed_max_val = float(max_integer + '.' + max_decimals[:idx])
        value_diff = (processed_max_val - processed_min_val) / (nr_labels + 1)
        inter_values = [round(processed_min_val + value_diff * i, idx) for i in range(nr_labels, 0, -1)]
        return [processed_max_val] + inter_values + [processed_min_val]

    def _prepare_data_slices(self, time_series_index):
        """
        Prepare data URL for retrieval with slices of timeSeries activity and Time-Line.
        :returns: [activity_urls], [timeline_urls]
                 Currently timeline_urls has just one value, as on client is loaded entirely anyway.
        """
        time_series_gid = time_series_index.gid
        activity_base_url = URLGenerator.build_url(self.stored_adapter.id, 'read_data_page_split', time_series_gid, "")
        time_urls = [SurfaceURLGenerator.build_h5_url(time_series_gid, 'read_time_page',
                                                      parameter="current_page=0;page_size=" +
                                                                str(time_series_index.data_length_1d))]
        return activity_base_url, time_urls

    def read_data_page_split(self, time_series_gid, from_idx, to_idx, step=None, specific_slices=None):
        with h5.h5_file_for_gid(time_series_gid) as time_series_h5:
            assert isinstance(time_series_h5, TimeSeriesH5)
            basic_result = time_series_h5.read_data_page(from_idx, to_idx, step, specific_slices)

            if not isinstance(time_series_h5, TimeSeriesSurfaceH5):
                return basic_result.tolist()
            surface_gid = time_series_h5.surface.load()

        result = []
        with h5.h5_file_for_gid(surface_gid) as surface_h5:
            assert isinstance(surface_h5, SurfaceH5)
            number_of_split_slices = surface_h5.number_of_split_slices.load()
            if number_of_split_slices <= 1:
                result.append(basic_result.tolist())
            else:
                for slice_number in range(surface_h5.number_of_split_slices):
                    start_idx, end_idx = surface_h5.get_slice_vertex_boundaries(slice_number)
                    result.append(basic_result[:, start_idx:end_idx].tolist())

        return result


class DualBrainViewerModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeries,
        label='Time Series'
    )

    projection_surface = DataTypeGidAttr(
        linked_datatype=Surface,
        required=False,
        label='Projection Surface',
        doc='A surface on which to project the results. When missing, the first EEGCap is taken. '
            'This parameter is ignored when InternalSensors measures.'
    )

    shell_surface = DataTypeGidAttr(
        linked_datatype=Surface,
        required=False,
        label='Shell Surface',
        doc='Wrapping surface over the internal sensors, to be displayed '
            'semi-transparently, for visual purposes only.'
    )


class DualBrainViewerForm(ABCAdapterForm):

    def __init__(self):
        super(DualBrainViewerForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(DualBrainViewerModel.time_series,
                                                    name='time_series',
                                                    conditions=self.get_filters())
        self.projection_surface = TraitDataTypeSelectField(DualBrainViewerModel.projection_surface,
                                                           name='projection_surface')
        self.shell_surface = TraitDataTypeSelectField(DualBrainViewerModel.shell_surface,
                                                      name='shell_surface')

    @staticmethod
    def get_view_model():
        return DualBrainViewerModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesIndex

    @staticmethod
    def get_input_name():
        return 'time_series'

    @staticmethod
    def get_filters():
        return FilterChain(
            fields=[FilterChain.datatype + '.time_series_type', FilterChain.datatype + '.has_surface_mapping'],
            operations=["in", "=="],
            values=[['TimeSeriesEEG', 'TimeSeriesSEEG', 'TimeSeriesMEG', 'TimeSeriesRegion'], True])


class DualBrainViewer(BrainViewer):
    """
    Visualizer merging Brain 3D display and EEG lines display.
    """
    _ui_name = "Brain Activity Viewer in 3D and 2D"
    _ui_subsection = "brain_dual"

    def get_form_class(self):
        return DualBrainViewerForm

    def populate_surface_fields(self, time_series_index):
        """
        Prepares the urls from which the client may read the data needed for drawing the surface.
        """

        if isinstance(time_series_index, TimeSeriesRegionIndex):
            BrainViewer.populate_surface_fields(self, time_series_index)
            return

        self.one_to_one_map = False
        self.region_map_gid = None
        self.connectivity_index = None

        if self.surface_index is None:
            eeg_cap = dao.get_generic_entity(SurfaceIndex, SurfaceTypesEnum.EEG_CAP_SURFACE.value, "surface_type")
            if len(eeg_cap) < 1:
                raise Exception("No EEG Cap Surface found for display!")
            self.surface_index = eeg_cap[0]

        self.surface_gid = self.surface_index.gid
        self.surface_h5 = h5.h5_file_for_index(self.surface_index)

    def retrieve_measure_points_params(self, time_series):

        if isinstance(time_series, TimeSeriesRegionIndex):
            return BrainViewer.retrieve_measure_points_params(self, time_series)

        sensors_index = dao.get_datatype_by_gid(time_series.fk_sensors_gid)
        self.measure_points_no = sensors_index.number_of_sensors

        if isinstance(time_series, TimeSeriesEEGIndex):
            return prepare_mapped_sensors_as_measure_points_params(sensors_index, self.surface_index,
                                                                   self.stored_adapter.id)

        return prepare_sensors_as_measure_points_params(sensors_index)

    def sensors_to_surface(self, sensors_gid, surface_to_map_gid):
        # Method needs to be defined on the adapter, to be called from JS
        return function_sensors_to_surface(sensors_gid, surface_to_map_gid)

    def launch(self, view_model):
        # type: (DualBrainViewerModel) -> dict

        time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.surface_index = None
        shell_surface_index = None

        if view_model.projection_surface:
            self.surface_index = self.load_entity_by_gid(view_model.projection_surface)

        if view_model.shell_surface:
            shell_surface_index = self.load_entity_by_gid(view_model.shell_surface)

        if isinstance(time_series_index, TimeSeriesSEEGIndex):
            shell_surface_index = ensure_shell_surface(self.current_project_id, shell_surface_index,
                                                       SurfaceTypesEnum.CORTICAL_SURFACE)

        params = BrainViewer.compute_parameters(self, time_series_index, shell_surface_index)
        eeg_monitor = EegMonitor()
        params.update(eeg_monitor.compute_parameters(time_series_index, is_extended_view=True))

        params['isOneToOneMapping'] = False
        params['brainViewerTemplate'] = 'view.html'

        if isinstance(time_series_index, TimeSeriesSEEGIndex):
            params['brainViewerTemplate'] = "internal_view.html"
            # Mark as None since we only display shell face and no point to load these as well
            params['urlVertices'] = None
            params['isSEEG'] = True

        return self.build_display_result("brain/extendedview", params,
                                         pages=dict(controlPage="brain/extendedcontrols",
                                                    channelsPage="commons/channel_selector.html"))
