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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import json
from tvb.adapters.visualizers.surface_view import prepare_shell_surface_urls
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import dao
from tvb.datatypes.sensors import Sensors, SensorsInternal, SensorsEEG, SensorsMEG
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.surfaces import EEGCap, CorticalSurface



def prepare_sensors_as_measure_points_params(sensors):
    """
    Returns urls from where to fetch the measure points and their labels
    """
    sensor_locations = ABCDisplayer.paths2url(sensors, 'locations')
    sensor_no = sensors.number_of_sensors
    sensor_labels = ABCDisplayer.paths2url(sensors, 'labels')

    return {'urlMeasurePoints': sensor_locations,
            'urlMeasurePointsLabels': sensor_labels,
            'noOfMeasurePoints': sensor_no,
            'minMeasure': 0,
            'maxMeasure': sensor_no,
            'urlMeasure': ''}



def prepare_mapped_sensors_as_measure_points_params(project_id, sensors, eeg_cap=None):
    """
    Compute sensors positions by mapping them to the ``eeg_cap`` surface
    If ``eeg_cap`` is not specified the mapping will use a default EEGCal DataType in current project.
    If no default EEGCap is found, return sensors as they are (not projected)

    :returns: dictionary to be used in Viewers for rendering measure_points
    :rtype: dict
    """

    if eeg_cap is None:
        eeg_cap = dao.try_load_last_entity_of_type(project_id, EEGCap)

    if eeg_cap:
        datatype_kwargs = json.dumps({'surface_to_map': eeg_cap.gid})
        sensor_locations = ABCDisplayer.paths2url(sensors, 'sensors_to_surface') + '/' + datatype_kwargs
        sensor_no = sensors.number_of_sensors
        sensor_labels = ABCDisplayer.paths2url(sensors, 'labels')

        return {'urlMeasurePoints': sensor_locations,
                'urlMeasurePointsLabels': sensor_labels,
                'noOfMeasurePoints': sensor_no,
                'minMeasure': 0,
                'maxMeasure': sensor_no,
                'urlMeasure': ''}

    return prepare_sensors_as_measure_points_params(sensors)



class SensorsViewer(ABCDisplayer):
    """
    Sensor visualizer - for visual inspecting of TVB Sensors DataTypes.
    """

    _ui_name = "Sensor Visualizer"
    _ui_subsection = "sensors"


    def get_input_tree(self):

        return [{'name': 'sensors', 'label': 'Sensors', 'type': Sensors, 'required': True,
                 'description': 'Internals sensors to view'},

                {'name': 'projection_surface', 'label': 'Projection Surface', 'type': Surface, 'required': False,
                 'description': 'A surface on which to project the results. When missing, the first EEGCap is taken'
                                'This parameter is ignored when InternalSensors are inspected'},

                {'name': 'shell_surface', 'label': 'Shell Surface', 'type': Surface, 'required': False,
                 'description': "Wrapping surface over the internal sensors, to be displayed "
                                "semi-transparently, for visual purposes only."}]


    def launch(self, sensors, projection_surface=None, shell_surface=None):
        """
        Prepare visualizer parameters.

        We support viewing all sensor types through a single viewer, so that a user doesn't need to
        go back to the data-page, for loading a different type of sensor.
        """
        if isinstance(sensors, SensorsInternal):
            return self._params_internal_sensors(sensors, shell_surface)

        if isinstance(sensors, SensorsEEG):
            return self._params_eeg_sensors(sensors, projection_surface, shell_surface)

        if isinstance(sensors, SensorsMEG):
            return self._params_meg_sensors(sensors, projection_surface, shell_surface)

        raise LaunchException("Unknown sensors type!")


    def _params_internal_sensors(self, internal_sensors, shell_surface=None):

        params = prepare_sensors_as_measure_points_params(internal_sensors)

        if shell_surface is None:
            shell_surface = dao.try_load_last_entity_of_type(self.current_project_id, CorticalSurface)
        params['shelfObject'] = prepare_shell_surface_urls(self.current_project_id, shell_surface)

        return self.build_display_result('sensors/sensors_internal', params,
                                         pages={'controlPage': 'sensors/sensors_controls'})


    def _params_eeg_sensors(self, eeg_sensors, eeg_cap=None, shell_surface=None):

        params = prepare_mapped_sensors_as_measure_points_params(self.current_project_id, eeg_sensors, eeg_cap)

        params.update({
            'shelfObject': prepare_shell_surface_urls(self.current_project_id, shell_surface),
            'urlVertices': '', 'urlTriangles': '', 'urlLines': '[]', 'urlNormals': ''
        })

        if eeg_cap is not None:
            params.update(self._compute_surface_params(eeg_cap))

        return self.build_display_result("sensors/sensors_eeg", params,
                                         pages={"controlPage": "sensors/sensors_controls"})


    def _params_meg_sensors(self, meg_sensors, projection_surface=None, shell_surface=None):

        params = prepare_sensors_as_measure_points_params(meg_sensors)

        params.update({
            'shelfObject': prepare_shell_surface_urls(self.current_project_id, shell_surface),
            'urlVertices': '', 'urlTriangles': '', 'urlLines': '[]', 'urlNormals': '',
            'boundaryURL': '', 'urlRegionMap': ''})

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
