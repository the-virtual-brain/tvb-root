# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: calin.pavel <calin.pavel@codemart.ro>
"""
import pytest
import os.path
import shutil
import zipfile
from contextlib import closing
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.adapters.exporters.exceptions import ExportException, InvalidExportDataException
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.simulator_h5 import SimulatorH5
from tvb.core.neocom import h5
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestExporters(TransactionalTestCase):
    """
    Test export functionality.
    """
    TVB_EXPORTER = "TVBExporter"
    CIFTI_EXPORTER = "CIFTIExporter"

    def transactional_setup_method(self):
        self.export_manager = ExportManager()
        self.test_user = TestFactory.create_user('Exporter_Tests_User1')
        self.test_project = TestFactory.create_project(self.test_user, 'Exporter_Tests_Project1')

    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        user = TestFactory.create_user('Exporter_Tests_User2')
        project = TestFactory.create_project(user, 'Exporter_Tests_Project2')
        FilesHelper().remove_project_structure(project.name)

        # Remove EXPORT folder
        export_folder = os.path.join(TvbProfile.current.TVB_STORAGE, ExportManager.EXPORT_FOLDER_NAME)
        if os.path.exists(export_folder):
            shutil.rmtree(export_folder)

    def test_get_exporters_for_data(self, dummy_datatype_index_factory):
        """
        Test retrieval of exporters that can be used for a given data.
        """
        datatype = dummy_datatype_index_factory()
        exporters = self.export_manager.get_exporters_for_data(datatype)

        # Only TVB export can export any type of data type
        assert 1, len(exporters) == "Incorrect number of exporters."

    def test_get_exporters_for_data_with_no_data(self):
        """
        Test retrieval of exporters when data == None.
        """
        with pytest.raises(InvalidExportDataException):
            self.export_manager.get_exporters_for_data(None)

    def test_tvb_export_of_simple_datatype(self, dummy_datatype_index_factory):
        """
        Test export of a data type which has no data stored on file system
        """
        datatype = dummy_datatype_index_factory()
        file_name, file_path, _ = self.export_manager.export_data(datatype, self.TVB_EXPORTER, self.test_project)

        assert file_name is not None, "Export process should return a file name"
        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file: %s on disk." % file_path

    def test_tvb_export_of_datatype_with_storage(self, dummy_datatype_index_factory):
        """
        Test export of a data type which has no data stored on file system
        """
        datatype = dummy_datatype_index_factory()
        file_name, file_path, _ = self.export_manager.export_data(datatype, self.TVB_EXPORTER, self.test_project)

        assert file_name is not None, "Export process should return a file name"
        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file: %s on disk." % file_path

    def test_tvb_export_for_datatype_group(self, datatype_group_factory):
        """
        This method checks export of a data type group
        """
        datatype_group = datatype_group_factory(project=self.test_project)
        file_name, file_path, _ = self.export_manager.export_data(datatype_group, self.TVB_EXPORTER, self.test_project)

        assert file_name is not None, "Export process should return a file name"
        assert file_path is not None, "Export process should return path to export file"
        assert os.path.exists(file_path), "Could not find export file: %s on disk." % file_path

        # Now check if the generated file is a correct ZIP file
        assert zipfile.is_zipfile(file_path), "Generated file is not a valid ZIP file"

        with closing(zipfile.ZipFile(file_path)) as zip_file:
            list_of_files = zip_file.namelist()

            count_datatypes = dao.count_datatypes_in_group(datatype_group.id)

            # Check if ZIP files contains files for data types
            assert count_datatypes == len(list_of_files)

    def test_export_with_invalid_data(self, dummy_datatype_index_factory):
        """
        Test scenarios when data provided to export method is invalid
        """
        # Test with no datatype
        with pytest.raises(InvalidExportDataException):
            self.export_manager.export_data(None, self.TVB_EXPORTER, self.test_project)
        # Test with no exporter 
        datatype = dummy_datatype_index_factory()
        with pytest.raises(ExportException):
            self.export_manager.export_data(datatype, None, self.test_project)

        # test with wrong exporter
        with pytest.raises(ExportException):
            self.export_manager.export_data(datatype, "wrong_exporter", self.test_project)

        # test with no project folder
        with pytest.raises(ExportException):
            self.export_manager.export_data(datatype, self.TVB_EXPORTER, None)

    def test_export_project_failure(self):
        """
        This method tests export of project with None data
        """
        with pytest.raises(ExportException):
            self.export_manager.export_project(None)

    def test_export_project(self, project_factory, user_factory):
        """
        Test export of a project
        """
        user = user_factory(username='test_user2')
        project = project_factory(user)
        export_file = self.export_manager.export_project(project)

        assert export_file is not None, "Export process should return path to export file"
        assert os.path.exists(export_file), "Could not find export file: %s on disk." % export_file
        # Now check if the generated file is a correct ZIP file
        assert zipfile.is_zipfile(export_file), "Generated file is not a valid ZIP file"

    def test_export_simulator_configuration(self, operation_factory, connectivity_factory):
        """
        Test export of a simulator configuration
        """
        operation = operation_factory()
        simulator = SimulatorAdapterModel()
        simulator.connectivity = connectivity_factory(4).gid

        burst_configuration = BurstConfiguration(self.test_project.id)
        burst_configuration.fk_simulation = operation.id
        burst_configuration.simulator_gid = simulator.gid.hex
        burst_configuration = dao.store_entity(burst_configuration)

        storage_path = FilesHelper().get_project_folder(self.test_project, str(operation.id))
        h5_path = h5.path_for(storage_path, SimulatorH5, simulator.gid)
        with SimulatorH5(h5_path) as h5_file:
            h5_file.store(simulator)

        export_file = self.export_manager.export_simulator_configuration(burst_configuration.id)

        assert export_file is not None, "Export process should return path to export file"
        assert os.path.exists(export_file), "Could not find export file: %s on disk." % export_file
        assert zipfile.is_zipfile(export_file), "Generated file is not a valid ZIP file"
