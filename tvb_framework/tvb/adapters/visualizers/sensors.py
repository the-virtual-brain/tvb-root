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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import json

from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.visualizers.surface_view import ensure_shell_surface, SurfaceURLGenerator
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer, URLGenerator
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.sensors import SensorsInternal, SensorsEEG, SensorsMEG, Sensors
from tvb.datatypes.surfaces import Surface, SurfaceTypesEnum

LOG = get_logger(__name__)


def prepare_sensors_as_measure_points_params(sensors):
    """
    Returns urls from where to fetch the measure points and their labels
    """
    sensor_locations = URLGenerator.build_h5_url(sensors.gid, 'get_locations')
    sensor_no = sensors.number_of_sensors
    sensor_labels = URLGenerator.build_h5_url(sensors.gid, 'get_labels')

    return {'urlMeasurePoints': sensor_locations,
            'urlMeasurePointsLabels': sensor_labels,
            'noOfMeasurePoints': sensor_no,
            'minMeasure': 0,
            'maxMeasure': sensor_no,
            'urlMeasure': ''}


def prepare_mapped_sensors_as_measure_points_params(sensors, eeg_cap=None, adapter_id=None):
    """
    Compute sensors positions by mapping them to the ``eeg_cap`` surface
    If ``eeg_cap`` is not specified the mapping will use a default EEGCal DataType in current project.
    If no default EEGCap is found, return sensors as they are (not projected)

    :returns: dictionary to be used in Viewers for rendering measure_points
    :rtype: dict
    """

    if eeg_cap:
        sensor_locations = URLGenerator.build_url(adapter_id, 'sensors_to_surface', sensors.gid,
                                                  parameter='surface_to_map_gid=' + eeg_cap.gid)
        sensor_no = sensors.number_of_sensors
        sensor_labels = URLGenerator.build_h5_url(sensors.gid, 'get_labels')

        return {'urlMeasurePoints': sensor_locations,
                'urlMeasurePointsLabels': sensor_labels,
                'noOfMeasurePoints': sensor_no,
                'minMeasure': 0,
                'maxMeasure': sensor_no,
                'urlMeasure': ''}

    return prepare_sensors_as_measure_points_params(sensors)


def function_sensors_to_surface(sensors_gid, surface_to_map_gid):
    """
    Map EEG sensors onto the head surface (skin-air).

    EEG sensor locations are typically only given on a unit sphere, that is,
    they are effectively only identified by their orientation with respect
    to a coordinate system. This method is used to map these unit vector
    sensor "locations" to a specific location on the surface of the skin.

    Assumes coordinate systems are aligned, i.e. common x,y,z and origin.

    """
    sensors_dt = h5.load_from_gid(sensors_gid)
    surface_dt = h5.load_from_gid(surface_to_map_gid)

    return sensors_dt.sensors_to_surface(surface_dt).tolist()


class SensorsViewerModel(ViewModel):
    sensors = DataTypeGidAttr(
        linked_datatype=Sensors,
        label='Sensors',
        doc='Internals sensors to view'
    )

    projection_surface = DataTypeGidAttr(
        linked_datatype=Surface,
        required=False,
        label='Projection Surface',
        doc='A surface on which to project the results. When missing, '
            'the first EEGCap is taken. This parameter is ignored when '
            'InternalSensors are inspected'
    )

    shell_surface = DataTypeGidAttr(
        linked_datatype=Surface,
        required=False,
        label='Shell Surface',
        doc='Wrapping surface over the internal sensors, to be displayed '
            'semi-transparently, for visual purposes only.'
    )


class SensorsViewerForm(ABCAdapterForm):

    def __init__(self):
        super(SensorsViewerForm, self).__init__()
        self.sensors = TraitDataTypeSelectField(SensorsViewerModel.sensors, name='sensors',
                                                conditions=self.get_filters())
        self.projection_surface = TraitDataTypeSelectField(SensorsViewerModel.projection_surface,
                                                           name='projection_surface')
        self.shell_surface = TraitDataTypeSelectField(SensorsViewerModel.shell_surface, name='shell_surface')

    @staticmethod
    def get_view_model():
        return SensorsViewerModel

    @staticmethod
    def get_required_datatype():
        return SensorsIndex

    @staticmethod
    def get_input_name():
        return 'sensors'

    @staticmethod
    def get_filters():
        return None


class SensorsViewer(ABCDisplayer):
    """
    Sensor visualizer - for visual inspecting of TVB Sensors DataTypes.
    """

    _ui_name = "Sensor Visualizer"
    _ui_subsection = "sensors"

    def get_form_class(self):
        return SensorsViewerForm

    def launch(self, view_model):
        # type: (SensorsViewerModel) -> dict
        """
        Prepare visualizer parameters.

        We support viewing all sensor types through a single viewer, so that a user doesn't need to
        go back to the data-page, for loading a different type of sensor.
        """
        sensors_index = self.load_entity_by_gid(view_model.sensors)
        shell_surface_index = None
        projection_surface_index = None

        if view_model.shell_surface:
            shell_surface_index = self.load_entity_by_gid(view_model.shell_surface)
        if view_model.projection_surface:
            projection_surface_index = self.load_entity_by_gid(view_model.projection_surface)

        if sensors_index.sensors_type == SensorsInternal.sensors_type.default:
            return self._params_internal_sensors(sensors_index, shell_surface_index)

        if sensors_index.sensors_type == SensorsEEG.sensors_type.default:
            return self._params_eeg_sensors(sensors_index, projection_surface_index, shell_surface_index)

        if sensors_index.sensors_type == SensorsMEG.sensors_type.default:
            return self._params_meg_sensors(sensors_index, projection_surface_index, shell_surface_index)

        raise LaunchException("Unknown sensors type!")

    def _params_internal_sensors(self, internal_sensors, shell_surface=None):

        params = prepare_sensors_as_measure_points_params(internal_sensors)

        shell_surface = ensure_shell_surface(self.current_project_id, shell_surface,
                                             SurfaceTypesEnum.CORTICAL_SURFACE.value)

        params['shellObject'] = self.prepare_shell_surface_params(shell_surface, SurfaceURLGenerator)

        return self.build_display_result('sensors/sensors_internal', params,
                                         pages={"controlPage": "sensors/sensors_controls"})

    def _params_eeg_sensors(self, eeg_sensors, eeg_cap=None, shell_surface=None):

        if eeg_cap is None:
            eeg_cap = ensure_shell_surface(self.current_project_id, eeg_cap, SurfaceTypesEnum.EEG_CAP_SURFACE.value)

        params = prepare_mapped_sensors_as_measure_points_params(eeg_sensors, eeg_cap, self.stored_adapter.id)

        shell_surface = ensure_shell_surface(self.current_project_id, shell_surface)

        params.update({
            'shellObject': self.prepare_shell_surface_params(shell_surface, SurfaceURLGenerator),
            'urlVertices': '', 'urlTriangles': '', 'urlLines': '[]', 'urlNormals': ''
        })

        if eeg_cap is not None:
            with h5.h5_file_for_gid(eeg_cap.gid) as eeg_cap_h5:
                params.update(self._compute_surface_params(eeg_cap_h5))

        return self.build_display_result("sensors/sensors_eeg", params,
                                         pages={"controlPage": "sensors/sensors_controls"})

    def _params_meg_sensors(self, meg_sensors, projection_surface=None, shell_surface=None):

        params = prepare_sensors_as_measure_points_params(meg_sensors)

        shell_surface = ensure_shell_surface(self.current_project_id, shell_surface)

        params.update({
            'shellObject': self.prepare_shell_surface_params(shell_surface, SurfaceURLGenerator),
            'urlVertices': '', 'urlTriangles': '', 'urlLines': '[]', 'urlNormals': '',
            'boundaryURL': '', 'urlRegionMap': ''})

        if projection_surface is not None:
            with h5.h5_file_for_gid(projection_surface.gid) as projection_surface_h5:
                params.update(self._compute_surface_params(projection_surface_h5))

        return self.build_display_result("sensors/sensors_eeg", params,
                                         pages={"controlPage": "sensors/sensors_controls"})

    @staticmethod
    def _compute_surface_params(surface_h5):
        rendering_urls = [json.dumps(url) for url in SurfaceURLGenerator.get_urls_for_rendering(surface_h5)]
        url_vertices, url_normals, url_lines, url_triangles, _ = rendering_urls

        return {'urlVertices': url_vertices,
                'urlTriangles': url_triangles,
                'urlLines': url_lines,
                'urlNormals': url_normals}

    def get_required_memory_size(self):
        return -1

    def sensors_to_surface(self, sensors_gid, surface_to_map_gid):
        # Method needs to be defined on the adapter, to be called from JS
        return function_sensors_to_surface(sensors_gid, surface_to_map_gid)
