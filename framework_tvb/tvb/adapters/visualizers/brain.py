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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import numpy
from tvb.adapters.visualizers.eeg_monitor import EegMonitor
from tvb.adapters.visualizers.surface_view import prepare_shell_surface_urls
from tvb.adapters.visualizers.sensors import prepare_sensors_as_measure_points_params
from tvb.adapters.visualizers.sensors import prepare_mapped_sensors_as_measure_points_params
from tvb.basic.filters.chain import FilterChain
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.surfaces import EEGCap, CorticalSurface
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.time_series import TimeSeries, TimeSeriesSurface, TimeSeriesSEEG, TimeSeriesEEG, TimeSeriesRegion


MAX_MEASURE_POINTS_LENGTH = 600



class BrainViewer(ABCDisplayer):
    """
    Interface between the 3D view of the Brain Cortical Surface and TVB framework.
    This viewer will build the required parameter dictionary that will be sent to the HTML / JS for further processing,
    having as end result a brain surface plus activity that will be displayed in 3D.
    """
    _ui_name = "Brain Activity Visualizer"
    PAGE_SIZE = 500


    def get_input_tree(self):
        return [{'name': 'time_series', 'label': 'Time Series (Region or Surface)',
                 'type': TimeSeries, 'required': True,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '.type',
                                                   FilterChain.datatype + '._has_surface_mapping'],
                                           operations=["in", "=="],
                                           values=[['TimeSeriesRegion', 'TimeSeriesSurface'], True])},

                {'name': 'shell_surface', 'label': 'Shell Surface', 'type': Surface, 'required': False,
                 'description': "Surface to be displayed semi-transparently, for visual purposes only."}]


    def get_required_memory_size(self, time_series, shell_surface=None):
        """
        Assume one page doesn't get 'dumped' in time and it is highly probably that
        two consecutive pages will be in the same time in memory.
        """
        overall_shape = time_series.read_data_shape()
        used_shape = (overall_shape[0] / (self.PAGE_SIZE * 2.0), overall_shape[1], overall_shape[2], overall_shape[3])
        return numpy.prod(used_shape) * 8.0


    def generate_preview(self, time_series, shell_surface=None, figure_size=None):
        """
        Generate the preview for the burst page
        """
        self.populate_surface_fields(time_series)

        url_vertices, url_normals, url_lines, url_triangles, url_region_map = self.surface.get_urls_for_rendering(True, self.region_map)
        params = self.retrieve_measure_points_prams(time_series)

        base_activity_url, time_urls = self._prepare_data_slices(time_series)
        min_val, max_val = time_series.get_min_max_values()

        if self.surface and self.region_map:
            boundary_url = self.surface.get_url_for_region_boundaries(self.region_map)
        else:
            boundary_url = ''

        params.update(urlVertices=json.dumps(url_vertices), urlTriangles=json.dumps(url_triangles),
                      urlLines=json.dumps(url_lines), urlNormals=json.dumps(url_normals),
                      urlRegionMap=json.dumps(url_region_map), urlRegionBoundaries=boundary_url,
                      base_activity_url=base_activity_url,
                      isOneToOneMapping=self.one_to_one_map, minActivity=min_val, maxActivity=max_val)

        normalization_factor = figure_size[0] / 800
        if figure_size[1] / 600 < normalization_factor:
            normalization_factor = figure_size[1] / 600
        params['width'] = figure_size[0] * normalization_factor
        params['height'] = figure_size[1] * normalization_factor

        return self.build_display_result("brain/portlet_preview", params)


    def launch(self, time_series, shell_surface=None):
        """
        Build visualizer's page.
        """
        params = self.compute_parameters(time_series, shell_surface)
        return self.build_display_result("brain/view", params, pages=dict(controlPage="brain/controls"))


    def populate_surface_fields(self, time_series):
        """
        To be overwritten for populating fields: one_to_one_map/connectivity/region_map/surface fields
        """

        self.one_to_one_map = isinstance(time_series, TimeSeriesSurface)

        if self.one_to_one_map:
            self.PAGE_SIZE /= 10
            self.surface = time_series.surface
            region_map = dao.get_generic_entity(RegionMapping, self.surface.gid, '_surface')
            if len(region_map) < 1:
                self.region_map = None
                self.connectivity = None
            else:
                self.region_map = region_map[0]
                self.connectivity = self.region_map.connectivity
        else:
            self.connectivity = time_series.connectivity
            self.region_map = time_series.region_mapping
            self.surface = self.region_map.surface


    def retrieve_measure_points_prams(self, time_series):
        """
        To be overwritten method, for retrieving the measurement points (region centers, EEG sensors).
        """
        if self.connectivity is None:
            self.measure_points_no = 0
            return {'urlMeasurePoints': [],
                    'urlMeasurePointsLabels': [],
                    'noOfMeasurePoints': 0}

        measure_points = ABCDisplayer.paths2url(self.connectivity, 'centres')
        measure_points_labels = ABCDisplayer.paths2url(self.connectivity, 'region_labels')
        self.measure_points_no = self.connectivity.number_of_regions

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

        url_vertices, url_normals, url_lines, url_triangles, url_region_map = self.surface.get_urls_for_rendering(True, self.region_map)
        hemisphere_chunk_mask = self.surface.get_slices_to_hemisphere_mask()

        params = self.retrieve_measure_points_prams(time_series)

        if not self.one_to_one_map and self.measure_points_no > MAX_MEASURE_POINTS_LENGTH:
            raise Exception("Max number of measure points " + str(MAX_MEASURE_POINTS_LENGTH) + " exceeded.")

        base_activity_url, time_urls = self._prepare_data_slices(time_series)
        min_val, max_val = time_series.get_min_max_values()
        legend_labels = self._compute_legend_labels(min_val, max_val)

        data_shape = time_series.read_data_shape()
        state_variables = time_series.labels_dimensions.get(time_series.labels_ordering[1], [])

        if self.surface and self.region_map:
            boundary_url = self.surface.get_url_for_region_boundaries(self.region_map)
        else:
            boundary_url = ''

        params.update(dict(title="Cerebral Activity: " + time_series.title, isOneToOneMapping=self.one_to_one_map,
                           urlVertices=json.dumps(url_vertices), urlTriangles=json.dumps(url_triangles),
                           urlLines=json.dumps(url_lines), urlNormals=json.dumps(url_normals),
                           urlRegionMap=json.dumps(url_region_map), base_activity_url=base_activity_url,
                           time=json.dumps(time_urls), minActivity=min_val, maxActivity=max_val,
                           legendLabels=legend_labels, labelsStateVar=state_variables,
                           labelsModes=range(data_shape[3]), extended_view=False,
                           shelfObject=prepare_shell_surface_urls(self.current_project_id, shell_surface),
                           biHemispheric=self.surface.bi_hemispheric,
                           hemisphereChunkMask=json.dumps(hemisphere_chunk_mask),
                           time_series=time_series, pageSize=self.PAGE_SIZE, urlRegionBoundaries=boundary_url,
                           measurePointsLabels=time_series.get_space_labels(),
                           measurePointsTitle=time_series.title))

        params.update(self.build_template_params_for_subselectable_datatype(time_series))

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


    def _prepare_data_slices(self, time_series):
        """
        Prepare data URL for retrieval with slices of timeSeries activity and Time-Line.
        :returns: [activity_urls], [timeline_urls]
                 Currently timeline_urls has just one value, as on client is loaded entirely anyway.
        """
        overall_shape = time_series.read_data_shape()

        activity_base_url = ABCDisplayer.VISUALIZERS_URL_PREFIX + time_series.gid
        time_urls = [self.paths2url(time_series, 'read_time_page',
                                    parameter="current_page=0;page_size=" + str(overall_shape[0]))]
        return activity_base_url, time_urls



class DualBrainViewer(BrainViewer):
    """
    Visualizer merging Brain 3D display and EEG lines display.
    """
    _ui_name = "Brain Activity Viewer in 3D and 2D"
    _ui_subsection = "brain_dual"


    def get_input_tree(self):

        return [{'name': 'time_series', 'label': 'Time Series', 'type': TimeSeries, 'required': True,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '.type',
                                                   FilterChain.datatype + '._has_surface_mapping'],
                                           operations=["in", "=="],
                                           values=[['TimeSeriesEEG', 'TimeSeriesSEEG',
                                                    'TimeSeriesMEG', 'TimeSeriesRegion'], True])},

                {'name': 'projection_surface', 'label': 'Projection Surface', 'type': Surface, 'required': False,
                 'description': 'A surface on which to project the results. When missing, the first EEGCap is taken'
                                'This parameter is ignored when InternalSensors measures.'},

                {'name': 'shell_surface', 'label': 'Shell Surface', 'type': Surface, 'required': False,
                 'description': "Wrapping surface over the internal sensors, to be displayed "
                                "semi-transparently, for visual purposes only."}]


    def populate_surface_fields(self, time_series):
        """
        Prepares the urls from which the client may read the data needed for drawing the surface.
        """

        if isinstance(time_series, TimeSeriesRegion):
            BrainViewer.populate_surface_fields(self, time_series)
            return

        self.one_to_one_map = False
        self.region_map = None
        self.connectivity = None

        if self.surface is None:
            eeg_cap = dao.get_generic_entity(EEGCap, "EEGCap", "type")
            if len(eeg_cap) < 1:
                raise Exception("No EEG Cap Surface found for display!")
            self.surface = eeg_cap[0]


    def retrieve_measure_points_prams(self, time_series):

        if isinstance(time_series, TimeSeriesRegion):
            return BrainViewer.retrieve_measure_points_prams(self, time_series)

        self.measure_points_no = time_series.sensors.number_of_sensors

        if isinstance(time_series, TimeSeriesEEG):
            return prepare_mapped_sensors_as_measure_points_params(self.current_project_id,
                                                                   time_series.sensors, self.surface)

        return prepare_sensors_as_measure_points_params(time_series.sensors)


    def launch(self, time_series, projection_surface=None, shell_surface=None):

        self.surface = projection_surface

        if isinstance(time_series, TimeSeriesSEEG) and shell_surface is None:
            shell_surface = dao.try_load_last_entity_of_type(self.current_project_id, CorticalSurface)

        params = BrainViewer.compute_parameters(self, time_series, shell_surface)
        params.update(EegMonitor().compute_parameters(time_series, is_extended_view=True))

        params['isOneToOneMapping'] = False
        params['brainViewerTemplate'] = 'view.html'

        if isinstance(time_series, TimeSeriesSEEG):
            params['brainViewerTemplate'] = "internal_view.html"
            # Mark as None since we only display shelf face and no point to load these as well
            params['urlVertices'] = None
            params['isSEEG'] = True

        return self.build_display_result("brain/extendedview", params,
                                         pages=dict(controlPage="brain/extendedcontrols",
                                                    channelsPage="commons/channel_selector.html"))

