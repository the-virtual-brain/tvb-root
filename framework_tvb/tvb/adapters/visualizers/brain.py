# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
from tvb.basic.filters.chain import FilterChain
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.surfaces import RegionMapping, EEGCap, FaceSurface
from tvb.datatypes.surfaces_data import SurfaceData
from tvb.datatypes.time_series import TimeSeries, TimeSeriesSurface, TimeSeriesSEEG, TimeSeriesRegion


MAX_MEASURE_POINTS_LENGTH = 235



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
                                                   FilterChain.datatype + '._nr_dimensions'],
                                           operations=["in", "=="],
                                           values=[['TimeSeriesRegion', 'TimeSeriesSurface'], 4]),
                 'description': 'Depending on the simulation length and your browser capabilities, you might experience'
                                ' after multiple runs, browser crashes. In such cases, it is recommended to empty the'
                                ' browser cache and try again. Sorry for the inconvenience.'}]


    def get_required_memory_size(self, time_series):
        """
        Return the required memory to run this algorithm.
        """
        overall_shape = time_series.read_data_shape()
        #Assume one page doesn't get 'dumped' in time and maybe two consecutive pages will be in the same
        #time in memory.
        used_shape = (overall_shape[0] / (self.PAGE_SIZE * 2.0), overall_shape[1], overall_shape[2], overall_shape[3])
        input_size = numpy.prod(used_shape) * 8.0
        return input_size


    def launch(self, time_series):
        """ Build visualizer's page. """
        params = self.compute_parameters(time_series)
        return self.build_display_result("brain/view", params, pages=dict(controlPage="brain/controls"))


    def generate_preview(self, time_series, figure_size=None):
        """ Generate the preview for the burst page """
        params = self.compute_preview_parameters(time_series)
        normalization_factor = figure_size[0] / 800
        if figure_size[1] / 600 < normalization_factor:
            normalization_factor = figure_size[1] / 600
        params['width'] = figure_size[0] * normalization_factor
        params['height'] = figure_size[1] * normalization_factor
        return self.build_display_result("brain/portlet_preview", params)


    def _populate_surface(self, time_series):

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
            region_map = dao.get_generic_entity(RegionMapping, self.connectivity.gid, '_connectivity')
            if len(region_map) < 1:
                raise Exception("No Mapping Surface found for display!")
            self.region_map = region_map[0]
            self.surface = self.region_map.surface


    def compute_preview_parameters(self, time_series):

        self._populate_surface(time_series)

        url_vertices, url_normals, url_lines, url_triangles, \
            alphas, alphas_indices = self.surface.get_urls_for_rendering(True, self.region_map)
        _, _, measure_points_no = self.retrieve_measure_points(time_series)
        min_val, max_val = time_series.get_min_max_values()

        return dict(urlVertices=json.dumps(url_vertices), urlTriangles=json.dumps(url_triangles),
                    urlLines=json.dumps(url_lines), urlNormals=json.dumps(url_normals),
                    alphas=json.dumps(alphas), alphas_indices=json.dumps(alphas_indices),
                    base_activity_url=ABCDisplayer.VISUALIZERS_URL_PREFIX + time_series.gid,
                    isOneToOneMapping=self.one_to_one_map, minActivity=min_val, maxActivity=max_val,
                    noOfMeasurePoints=measure_points_no)


    @staticmethod
    def get_shell_surface_urls(shell_surface=None, project_id=0):

        if shell_surface is None:
            shell_surface = dao.get_values_of_datatype(project_id, FaceSurface)[0]

            if not shell_surface:
                raise Exception('No face object found in database.')

            shell_surface = ABCDisplayer.load_entity_by_gid(shell_surface[0][2])

        face_vertices, face_normals, _, face_triangles = shell_surface.get_urls_for_rendering()
        return json.dumps([face_vertices, face_normals, face_triangles])


    def compute_parameters(self, time_series, shell_surface=None):
        """
        Create the required parameter dictionary for the HTML/JS viewer.

        :rtype: `dict`
        :raises Exception: when
                    * number of measure points exceeds the maximum allowed
                    * a Face object cannot be found in database

        """
        self._populate_surface(time_series)

        url_vertices, url_normals, url_lines, url_triangles, \
            alphas, alphas_indices = self.surface.get_urls_for_rendering(True, self.region_map)
        hemisphere_chunk_mask = self.surface.get_slices_to_hemisphere_mask()

        measure_points, measure_points_labels, measure_points_no = self.retrieve_measure_points(time_series)
        if not self.one_to_one_map and measure_points_no > MAX_MEASURE_POINTS_LENGTH:
            raise Exception("Max number of measure points " + str(MAX_MEASURE_POINTS_LENGTH) + " exceeded.")

        base_activity_url, time_urls = self._prepare_data_slices(time_series)
        min_val, max_val = time_series.get_min_max_values()
        legend_labels = self._compute_legend_labels(min_val, max_val)

        face_object = BrainViewer.get_shell_surface_urls(shell_surface, self.current_project_id)

        data_shape = time_series.read_data_shape()
        state_variables = time_series.labels_dimensions.get(time_series.labels_ordering[1], [])

        if self.surface and self.region_map:
            boundary_url = self.surface.get_url_for_region_boundaries(self.region_map)
        else:
            boundary_url = ''

        retu = dict(title=self._get_subtitle(time_series), isOneToOneMapping=self.one_to_one_map,
                    urlVertices=json.dumps(url_vertices), urlTriangles=json.dumps(url_triangles),
                    urlLines=json.dumps(url_lines), urlNormals=json.dumps(url_normals),
                    urlMeasurePointsLabels=measure_points_labels, measure_points=measure_points,
                    noOfMeasurePoints=measure_points_no, alphas=json.dumps(alphas),
                    alphas_indices=json.dumps(alphas_indices), base_activity_url=base_activity_url,
                    time=json.dumps(time_urls), minActivity=min_val, maxActivity=max_val,
                    minActivityLabels=legend_labels, labelsStateVar=state_variables, labelsModes=range(data_shape[3]),
                    extended_view=False, shelfObject=face_object,
                    biHemispheric=self.surface.bi_hemispheric, hemisphereChunkMask=json.dumps(hemisphere_chunk_mask),
                    time_series=time_series, pageSize=self.PAGE_SIZE, urlRegionBoundaries=boundary_url,
                    measurePointsLabels=time_series.get_space_labels(),
                    measurePointsTitle=time_series.title)

        retu.update(self.build_template_params_for_subselectable_datatype(time_series))

        return retu

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
            for i in xrange(1, len(vertices_indexes)):
                if vertices_indexes[i][0] == vertices_indexes[i][1]:
                    this_mappings.append(vertices_indexes[i][0])
                else:
                    for index in xrange(vertices_indexes[i][0], vertices_indexes[i][1] + 1):
                        this_mappings.append(index)
            prepared_mappings.append(this_mappings)
        return prepared_mappings


    def retrieve_measure_points(self, time_series):
        """
        To be overwritten method, for retrieving the measurement points (region centers, EEG sensors).
        """
        if isinstance(time_series, TimeSeriesSurface):
            return [], [], 0
        measure_points = ABCDisplayer.paths2url(time_series.connectivity, 'centres')
        measure_points_labels = ABCDisplayer.paths2url(time_series.connectivity, 'region_labels')
        measure_points_no = time_series.connectivity.number_of_regions
        return measure_points, measure_points_labels, measure_points_no


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
        inter_values = [round(processed_min_val + value_diff * i, idx) for i in xrange(nr_labels, 0, -1)]
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


    def _get_subtitle(self, time_series):
        return "Cerebral Activity: " + time_series.title



class BrainEEG(BrainViewer):
    """
    Visualizer merging Brain 3D display and EEG lines display.
    """
    _ui_name = "Brain EEG Activity in 3D and 2D"
    _ui_subsection = "brain_dual"


    def get_input_tree(self):
        
        return [{'name': 'surface_activity', 'label': 'Time Series (EEG or MEG)',
                 'type': TimeSeries, 'required': True,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '.type'],
                                           operations=["in"],
                                           values=[['TimeSeriesEEG', 'TimeSeriesMEG']]),
                 'description': 'Depending on the simulation length and your browser capabilities, you might experience'
                                ' after multiple runs, browser crashes. In such cases, it is recommended to empty the'
                                ' browser cache and try again. Sorry for the inconvenience.'},
                {'name': 'eeg_cap', 'label': 'EEG Cap',
                 'type': EEGCap, 'required': False,
                 'description': 'The EEG Cap surface on which to display the results!'}]


    @staticmethod
    def get_sensor_measure_points(sensors):
        """
            Returns urls from where to fetch the measure points and their labels
        """
        measure_points = ABCDisplayer.paths2url(sensors, 'locations')
        measure_points_no = sensors.number_of_sensors
        measure_points_labels = ABCDisplayer.paths2url(sensors, 'labels')
        return measure_points, measure_points_labels, measure_points_no


    @staticmethod
    def compute_sensor_surfacemapped_measure_points(project_id, sensors, eeg_cap=None):
        """
        Compute sensors positions by mapping them to the ``eeg_cap`` surface
        If ``eeg_cap`` is not specified the mapping will use a default.
        It returns a url from where to fetch the positions
        If no default is available it returns None
        :returns: measure points, measure points labels, measure points number
        :rtype: tuple
        """

        if eeg_cap is None:
            eeg_cap = dao.get_values_of_datatype(project_id, EEGCap)[0]
            if eeg_cap:
                eeg_cap = ABCDisplayer.load_entity_by_gid(eeg_cap[-1][2])

        if eeg_cap:
            datatype_kwargs = json.dumps({'surface_to_map': eeg_cap.gid})
            measure_points = ABCDisplayer.paths2url(sensors, 'sensors_to_surface') + '/' + datatype_kwargs
            measure_points_no = sensors.number_of_sensors
            measure_points_labels = ABCDisplayer.paths2url(sensors, 'labels')
            return measure_points, measure_points_labels, measure_points_no


    def retrieve_measure_points(self, surface_activity):
        measure_point_info = BrainEEG.compute_sensor_surfacemapped_measure_points(self.current_project_id,
                                                                                  surface_activity.sensors,
                                                                                  self.surface)
        if measure_point_info is None:
            measure_point_info = BrainEEG.get_sensor_measure_points(surface_activity.sensors)
        return measure_point_info


    def launch(self, surface_activity, eeg_cap=None, shell_surface=None):
        """
        Overwrite Brain Visualizer launch and extend functionality,
        by adding a Monitor set of parameters near.
        """
        self.surface = eeg_cap
        params = BrainViewer.compute_parameters(self, surface_activity, shell_surface)
        params.update(EegMonitor().compute_parameters(surface_activity, is_extended_view=True))
        params['brainViewerTemplate'] = 'view.html'
        return self.build_display_result("brain/extendedview", params,
                                         pages=dict(controlPage="brain/extendedcontrols",
                                                    channelsPage="commons/channel_selector.html"))


    def _populate_surface(self, time_series):
        """
        Prepares the urls from which the client may read the data needed for drawing the surface.
        """
        self.one_to_one_map = False
        self.region_map = None
        self.connectivity = None
        if self.surface is None:
            eeg_cap = dao.get_generic_entity(EEGCap, "EEGCap", "type")
            if len(eeg_cap) < 1:
                raise Exception("No EEG Cap Surface found for display!")
            self.surface = eeg_cap[0]



class BrainSEEG(BrainEEG):
    """
    Visualizer merging Brain 3D display and MEG lines display.
    """
    _ui_name = "Brain SEEG Activity in 3D and 2D"
    _ui_subsection = "brain_dual"


    def get_input_tree(self):
        return [{'name': 'surface_activity', 'label': 'SEEG activity',
                 'type': TimeSeriesSEEG, 'required': True,
                 'description': 'Results after SEEG Monitor are expected!'},
                {'name': 'shell_surface', 'label': 'Surface',
                 'type': SurfaceData, 'required': False,
                 'description': "Surface to be displayed semi-transparently, for visual purposes only."}]


    def retrieve_measure_points(self, surface_activity):
        return BrainEEG.get_sensor_measure_points(surface_activity.sensors)


    def launch(self, surface_activity, shell_surface=None):
        result_params = BrainEEG.launch(self, surface_activity, shell_surface=shell_surface)
        result_params['brainViewerTemplate'] = "internal_view.html"
        # Mark as None since we only display shelf face and no point to load these as well
        result_params['urlVertices'] = None
        result_params['isSEEG'] = True
        return result_params
        


class BrainRegionDual(BrainViewer):
    """
    Visualizer merging Brain 3D display and animated time series display
    """
    _ui_name = "Brain Regions Activity in 3D and 2D"
    _ui_subsection = "brain_dual"


    def get_input_tree(self):
        return [{'name': 'region_activity', 'label': 'Time Series Region',
                 'type': TimeSeriesRegion, 'required': True,
                 'description': 'dual view'}]


    def launch(self, region_activity):
        """
        Overwrite Brain Visualizer launch and extend functionality,
        by adding a Monitor set of parameters near.
        """
        params = BrainViewer.compute_parameters(self, region_activity)
        params.update(EegMonitor().compute_parameters(region_activity, is_extended_view=True))
        params['biHemispheric'] = False
        params['isOneToOneMapping'] = False
        params['brainViewerTemplate'] = 'view.html'
        params['title'] = self._get_subtitle(region_activity)
        return self.build_display_result("brain/extendedview", params,
                                         pages=dict(controlPage="brain/extendedcontrols",
                                                    channelsPage="commons/channel_selector.html"))
