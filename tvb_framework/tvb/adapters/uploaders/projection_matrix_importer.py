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
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitUploadField, StrField, TraitDataTypeSelectField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.datatypes.projections import *
from tvb.datatypes.projections import ProjectionMatrix
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG, Sensors
from tvb.datatypes.surfaces import Surface

DEFAULT_DATASET_NAME = "ProjectionMatrix"


def determine_projection_type(sensors):
    # type: (SensorsIndex) -> str
    if sensors.sensors_type == SensorsEEG.sensors_type.default:
        projection_matrix_type = ProjectionSurfaceEEG.projection_type.default
    elif sensors.sensors_type == SensorsMEG.sensors_type.default:
        projection_matrix_type = ProjectionSurfaceMEG.projection_type.default
    else:
        projection_matrix_type = ProjectionSurfaceSEEG.projection_type.default

    return projection_matrix_type


class ProjectionMatrixImporterModel(UploaderViewModel):
    projection_file = Str(
        label='Projection matrix file (.mat or .npy format)',
        doc='Expected a file containing projection matrix (one vector of length '
            'number of surface vertices nd values in the sensors range).'
    )

    dataset_name = Attr(
        field_type=str,
        required=False,
        default=DEFAULT_DATASET_NAME,
        label='Matlab dataset name',
        doc='Name of the MATLAB dataset where data is stored. Required only for .mat files'
    )

    surface = DataTypeGidAttr(
        linked_datatype=Surface,
        label='Brain Cortical Surface',
        doc='The Brain Surface used by the uploaded projection matrix.'
    )

    sensors = DataTypeGidAttr(
        linked_datatype=Sensors,
        label='Sensors',
        doc='The Sensors used in for current projection.'
    )


class ProjectionMatrixImporterForm(ABCUploaderForm):

    def __init__(self):
        super(ProjectionMatrixImporterForm, self).__init__()
        self.projection_file = TraitUploadField(ProjectionMatrixImporterModel.projection_file, ('.mat', '.npy'),
                                                'projection_file')
        self.dataset_name = StrField(ProjectionMatrixImporterModel.dataset_name, name='dataset_name')
        surface_conditions = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=['=='],
                                         values=['Cortical Surface'])
        self.surface = TraitDataTypeSelectField(ProjectionMatrixImporterModel.surface, name='surface',
                                                conditions=surface_conditions)
        self.sensors = TraitDataTypeSelectField(ProjectionMatrixImporterModel.sensors, name='sensors')

    @staticmethod
    def get_view_model():
        return ProjectionMatrixImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'projection_file': ('.mat', '.npy')
        }


class ProjectionMatrixSurfaceEEGImporter(ABCUploader):
    """
    Upload ProjectionMatrix Cortical Surface -> EEG/MEG/SEEG Sensors from a MAT or NPY file.
    """

    _ui_name = "Gain Matrix for Sensors"
    _ui_description = "Upload a Projection Matrix between a Brain Cortical Surface and EEG/MEG Sensors."
    logger = get_logger(__name__)

    def get_form_class(self):
        return ProjectionMatrixImporterForm

    def get_output(self):
        return [ProjectionMatrixIndex]

    def launch(self, view_model):
        # type: (ProjectionMatrixImporterModel) -> [ProjectionMatrixIndex]
        """
        Creates ProjectionMatrix entity from uploaded data.

        :raises LaunchException: when
                    * no projection_file or sensors are specified
                    * the dataset is invalid
                    * number of sensors is different from the one in dataset
        """
        if view_model.projection_file is None:
            raise LaunchException("Please select MATLAB file which contains data to import")

        if view_model.sensors is None:
            raise LaunchException("No sensors selected. Please initiate upload again and select one.")

        if view_model.surface is None:
            raise LaunchException("No source selected. Please initiate upload again and select a source.")

        sensors_ht = self.load_traited_by_gid(view_model.sensors)
        expected_sensors_shape = sensors_ht.number_of_sensors

        self.logger.debug("Reading projection matrix from uploaded file...")
        if view_model.projection_file.endswith(".mat"):
            projection_data = self.read_matlab_data(view_model.projection_file, view_model.dataset_name)
        else:
            projection_data = self.read_list_data(view_model.projection_file)

        if projection_data is None or len(projection_data) == 0:
            raise LaunchException("Invalid (empty) dataset...")

        if projection_data.shape[0] != expected_sensors_shape:
            raise LaunchException("Invalid Projection Matrix shape[0]: %d Expected: %d" % (projection_data.shape[0],
                                                                                           expected_sensors_shape))

        surface_idx = self.load_entity_by_gid(view_model.surface)
        expected_surface_shape = surface_idx.number_of_vertices

        if projection_data.shape[1] != expected_surface_shape:
            raise LaunchException("Invalid Projection Matrix shape[1]: %d Expected: %d" % (projection_data.shape[1],
                                                                                           expected_surface_shape))

        surface_ht = h5.load_from_index(surface_idx)
        projection_matrix_type = determine_projection_type(sensors_ht)
        projection_matrix = ProjectionMatrix(sources=surface_ht, sensors=sensors_ht,
                                             projection_type=projection_matrix_type,
                                             projection_data=projection_data)
        return self.store_complete(projection_matrix)
