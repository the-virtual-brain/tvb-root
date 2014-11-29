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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import json
from tvb.adapters.visualizers.brain import BrainEEG, BrainViewer
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.adapters.exceptions import LaunchException
from tvb.datatypes.sensors_data import SensorsData
from tvb.datatypes.sensors import SensorsInternal, SensorsEEG, SensorsMEG
from tvb.datatypes.surfaces_data import SurfaceData



class SensorsViewer(ABCDisplayer):
    """
    Sensor visualizer - for visual inspecting of TVB Sensors DataTypes.
    """

    _ui_name = "Sensor Visualizer"
    _ui_subsection = "sensors"


    def get_input_tree(self):

        return [{'name': 'sensors', 'label': 'Sensors', 'type': SensorsData, 'required': True,
                 'description': 'Internals sensors to view'},

                {'name': 'projection_surface', 'label': 'Projection Surface', 'type': SurfaceData, 'required': False,
                 'description': 'A surface on which to project the results. When missing, the first EEGCap is taken'
                                'This parameter is ignored when InternalSensors are inspected'},

                {'name': 'shell_surface', 'label': 'Shell Surface', 'type': SurfaceData, 'required': False,
                 'description': "Wrapping surface over the internal sensors, to be displayed "
                                "semi-transparently, for visual purposes only."}]


    def launch(self, sensors, projection_surface=None, shell_surface=None):
        """
        Prepare visualizer parameters
        """
        if isinstance(sensors, SensorsInternal):
            return self._params_internal_sensors(sensors, shell_surface)

        if isinstance(sensors, SensorsEEG):
            return self._params_eeg_sensors(sensors, projection_surface, shell_surface)

        if isinstance(sensors, SensorsMEG):
            return self._params_meeg_sensors(sensors, projection_surface, shell_surface)

        raise LaunchException("Unknown sensors type!")


    def _params_internal_sensors(self, internal_sensors, shell_surface=None):

        sensor_locations, sensor_labels, sensor_no = BrainEEG.get_sensor_measure_points(internal_sensors)

        params = {
            'shelfObject': BrainViewer.get_shell_surface_urls(shell_surface, self.current_project_id),
            'urlMeasurePoints': sensor_locations,
            'urlMeasurePointsLabels': sensor_labels,
            'noOfMeasurePoints': sensor_no,
            'minMeasure': 0,
            'maxMeasure': sensor_no,
            'urlMeasure': ''
        }

        return self.build_display_result('sensors/sensors_internal', params,
                                         pages={'controlPage': 'sensors/sensors_controls'})


    def _params_eeg_sensors(self, eeg_sensors, eeg_cap=None, shell_surface=None):

        measure_points_info = BrainEEG.compute_sensor_surfacemapped_measure_points(self.current_project_id,
                                                                                   eeg_sensors, eeg_cap)

        measure_points_nr = measure_points_info[2]
        params = {
            'shelfObject': BrainViewer.get_shell_surface_urls(shell_surface, self.current_project_id),
            'urlVertices': '', 'urlTriangles': '',
            'urlLines': '[]', 'urlNormals': '',
            'boundaryURL': '', 'urlAlphas': '', 'urlAlphasIndices': '',
            'urlMeasurePoints': measure_points_info[0],
            'urlMeasurePointsLabels': measure_points_info[1],
            'noOfMeasurePoints': measure_points_nr,
            'minMeasure': 0,
            'maxMeasure': measure_points_nr,
            'urlMeasure': ''
        }

        if eeg_cap is not None:
            params.update(self._compute_surface_params(eeg_cap))

        return self.build_display_result("sensors/sensors_eeg", params,
                                         pages={"controlPage": "sensors/sensors_controls"})


    def _params_meeg_sensors(self, meg_sensors, projection_surface=None, shell_surface=None):

        sensor_locations, sensor_labels, sensor_no = BrainEEG.get_sensor_measure_points(meg_sensors)

        params = {
            'shelfObject': BrainViewer.get_shell_surface_urls(shell_surface, self.current_project_id),
            'urlVertices': '', 'urlTriangles': '',
            'urlLines': '[]', 'urlNormals': '',
            'boundaryURL': '', 'urlAlphas': '', 'urlAlphasIndices': '',
            'urlMeasurePoints': sensor_locations,
            'urlMeasurePointsLabels': sensor_labels,
            'noOfMeasurePoints': sensor_no,
            'minMeasure': 0,
            'maxMeasure': sensor_no,
            'urlMeasure': ''
        }

        if projection_surface is not None:
            params.update(self._compute_surface_params(projection_surface))

        return self.build_display_result("sensors/sensors_eeg", params,
                                         pages={"controlPage": "sensors/sensors_controls"})


    @staticmethod
    def _compute_surface_params(surface):
        rendering_urls = [json.dumps(url) for url in surface.get_urls_for_rendering()]
        url_vertices, url_normals, url_lines, url_triangles = rendering_urls
        return {'urlVertices': url_vertices,
                'urlTriangles': url_triangles,
                'urlLines': url_lines,
                'urlNormals': url_normals}


    def get_required_memory_size(self):
        return -1

