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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""
import os
import tvb_data
import shutil
import pytest
from tvb.adapters.datatypes.db.mapped_value import ValueWrapperIndex
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.basic.profile import TvbProfile
from tvb.core.entities.storage import dao
from tvb.core.entities.load import try_get_last_datatype
from tvb.core.services.import_service import ImportService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.exceptions import ProjectImportException
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.core.base_testcase import BaseTestCase


class TestImportService(BaseTestCase):
    """
    This class contains tests for the tvb.core.services.import_service module.
    """

    def setup_method(self):
        """
        Reset the database before each test.
        """
        self.import_service = ImportService()
        self.project_service = ProjectService()
        self.zip_path = None

    def teardown_method(self):
        """
        Reset the database when test is done.
        """
        # Delete TEMP folder
        if os.path.exists(TvbProfile.current.TVB_TEMP_FOLDER):
            shutil.rmtree(TvbProfile.current.TVB_TEMP_FOLDER)

        # Delete folder where data was exported
        if self.zip_path and os.path.exists(self.zip_path):
            shutil.rmtree(os.path.split(self.zip_path)[0])

        self.delete_project_folders()

    def test_import_export(self, user_factory, project_factory, value_wrapper_factory):
        """
        Test the import/export mechanism for a project structure.
        The project contains the following data types: Connectivity, Surface, MappedArray and ValueWrapper.
        """
        test_user = user_factory()
        test_project = project_factory(test_user, "TestImportExport")
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(test_user, test_project, zip_path)
        value_wrapper = value_wrapper_factory(test_user, test_project)

        result = self.get_all_datatypes()
        expected_results = {}
        for one_data in result:
            expected_results[one_data.gid] = (one_data.module, one_data.type)

        # Export project as ZIP
        self.zip_path = ExportManager().export_project(test_project)
        assert self.zip_path is not None, "Exported file is none"

        # Remove the original project
        self.project_service.remove_project(test_project.id)
        result, lng_ = self.project_service.retrieve_projects_for_user(test_user.id)
        assert 0 == len(result), "Project Not removed!"
        assert 0 == lng_, "Project Not removed!"

        # Now try to import again project
        self.import_service.import_project_structure(self.zip_path, test_user.id)
        result = self.project_service.retrieve_projects_for_user(test_user.id)[0]
        assert len(result) == 1, "There should be only one project."
        assert result[0].name == "GeneratedProject", "The project name is not correct."
        assert result[0].description == "test_desc", "The project description is not correct."
        test_project = result[0]

        count_operations = dao.get_filtered_operations(test_project.id, None, is_count=True)

        # 1 op. - import conn; 2 op. - BCT Analyzer
        assert 2 == count_operations, "Invalid ops number after export and import !"
        for gid in expected_results:
            datatype = dao.get_datatype_by_gid(gid)
            assert datatype.module == expected_results[gid][0], 'DataTypes not imported correctly'
            assert datatype.type == expected_results[gid][1], 'DataTypes not imported correctly'
        # check the value wrapper
        new_val = try_get_last_datatype(test_project.id, ValueWrapperIndex)
        assert value_wrapper.data_value == new_val.data_value, "Data value incorrect"
        assert value_wrapper.data_type == new_val.data_type, "Data type incorrect"
        assert value_wrapper.data_name == new_val.data_name, "Data name incorrect"

    def test_import_export_existing(self, user_factory, project_factory):
        """
        Test the import/export mechanism for a project structure.
        The project contains the following data types: Connectivity, Surface, MappedArray and ValueWrapper.
        """
        test_user = user_factory()
        test_project = project_factory(test_user, "TestImportExport2")
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(test_user, test_project, zip_path)

        count_operations = dao.get_filtered_operations(test_project.id, None, is_count=True)
        assert 1 == count_operations, "Invalid ops before export!"

        self.zip_path = ExportManager().export_project(test_project)
        assert self.zip_path is not None, "Exported file is none"

        with pytest.raises(ProjectImportException):
            self.import_service.import_project_structure(self.zip_path, test_user.id)
