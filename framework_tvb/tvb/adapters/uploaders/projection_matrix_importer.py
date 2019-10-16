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
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import uuid
from tvb.basic.filters.chain import FilterChain
from tvb.adapters.uploaders.abcuploader import ABCUploader, ABCUploaderForm
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import LaunchException
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG
from tvb.datatypes.projections import ProjectionSurfaceEEG, ProjectionSurfaceMEG, ProjectionSurfaceSEEG
from tvb.core.entities.file.datatypes.projections_h5 import ProjectionMatrixH5
from tvb.core.entities.model.datatypes.projections import ProjectionMatrixIndex
from tvb.core.entities.model.datatypes.sensors import SensorsIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.neotraits._forms import UploadField, SimpleStrField, DataTypeSelectField
from tvb.interfaces.neocom._h5loader import DirLoader

DEFAULT_DATASET_NAME = "ProjectionMatrix"


def determine_projection_type(sensors):
    if sensors.sensors_type == SensorsEEG.sensors_type.default:
        projection_matrix_type = ProjectionSurfaceEEG.projection_type.default
    elif sensors.sensors_type == SensorsMEG.sensors_type.default:
        projection_matrix_type = ProjectionSurfaceMEG.projection_type.default
    else:
        projection_matrix_type = ProjectionSurfaceSEEG.projection_type.default

    return projection_matrix_type


class ProjectionMatrixImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None):
        super(ProjectionMatrixImporterForm, self).__init__(prefix, project_id)
        self.projection_file = UploadField('.mat, .npy', self, name='projection_file', required=True,
                                           label='Projection matrix file (.mat or .npy format)',
                                           doc='Expected a file containing projection matrix (one vector of length '
                                               'number of surface vertices nd values in the sensors range).')
        self.dataset_name = SimpleStrField(self, name='dataset_name', default=DEFAULT_DATASET_NAME,
                                           label='Matlab dataset name',
                                           doc='Name of the MATLAB dataset where data is stored. Required only for .mat files')
        surface_conditions = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=['=='],
                                         values=['Cortical Surface'])
        self.surface = DataTypeSelectField(SurfaceIndex, self, name='surface', required=True,
                                           conditions=surface_conditions, label='Brain Cortical Surface',
                                           doc='The Brain Surface used by the uploaded projection matrix.')
        self.sensors = DataTypeSelectField(SensorsIndex, self, name='sensors', required=True, label='Sensors',
                                           doc='The Sensors used in for current projection.')


class ProjectionMatrixSurfaceEEGImporter(ABCUploader):
    """
    Upload ProjectionMatrix Cortical Surface -> EEG/MEG/SEEG Sensors from a MAT or NPY file.
    """

    _ui_name = "Gain Matrix for Sensors"
    _ui_description = "Upload a Projection Matrix between a Brain Cortical Surface and EEG/MEG Sensors."
    logger = get_logger(__name__)

    form = None

    def get_input_tree(self): return None

    def get_upload_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return ProjectionMatrixImporterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [ProjectionSurfaceEEG, ProjectionSurfaceMEG, ProjectionSurfaceSEEG]


    def launch(self, projection_file, surface, sensors, dataset_name=DEFAULT_DATASET_NAME):
        """
        Creates ProjectionMatrix entity from uploaded data.

        :raises LaunchException: when
                    * no projection_file or sensors are specified
                    * the dataset is invalid
                    * number of sensors is different from the one in dataset
        """
        if projection_file is None:
            raise LaunchException("Please select MATLAB file which contains data to import")

        if sensors is None:
            raise LaunchException("No sensors selected. Please initiate upload again and select one.")

        if surface is None:
            raise LaunchException("No source selected. Please initiate upload again and select a source.")
        expected_shape = surface.number_of_vertices

        self.logger.debug("Reading projection matrix from uploaded file...")
        if projection_file.endswith(".mat"):
            eeg_projection_data = self.read_matlab_data(projection_file, dataset_name)
        else:
            eeg_projection_data = self.read_list_data(projection_file)

        if eeg_projection_data is None or len(eeg_projection_data) == 0:
            raise LaunchException("Invalid (empty) dataset...")

        if eeg_projection_data.shape[0] != sensors.number_of_sensors:
            raise LaunchException("Invalid Projection Matrix shape[0]: %d Expected: %d" % (eeg_projection_data.shape[0],
                                                                                           sensors.number_of_sensors))

        if eeg_projection_data.shape[1] != expected_shape:
            raise LaunchException("Invalid Projection Matrix shape[1]: %d Expected: %d" % (eeg_projection_data.shape[1],
                                                                                           expected_shape))

        projection_matrix_type = determine_projection_type(sensors)
        projection_matrix_idx = ProjectionMatrixIndex()
        projection_matrix_idx.source = surface
        projection_matrix_idx.source_id = surface.id
        projection_matrix_idx.sensors = sensors
        projection_matrix_idx.sensors_id = sensors.id
        projection_matrix_idx.projection_type = projection_matrix_type

        loader = DirLoader(self.storage_path)
        projection_matrix_path = loader.path_for(ProjectionMatrixH5, projection_matrix_idx.gid)

        with ProjectionMatrixH5(projection_matrix_path) as projection_matrix_h5:
            projection_matrix_h5.projection_type.store(projection_matrix_type)
            projection_matrix_h5.projection_data.store(eeg_projection_data)
            projection_matrix_h5.sources.store(uuid.UUID(surface.gid))
            projection_matrix_h5.sensors.store(uuid.UUID(sensors.gid))
            projection_matrix_h5.gid.store(uuid.UUID(projection_matrix_idx.gid))

        return [projection_matrix_idx]
