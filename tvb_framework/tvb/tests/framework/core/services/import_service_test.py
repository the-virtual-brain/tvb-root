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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import os
import pytest
import tvb_data
from PIL import Image
from time import sleep
from tvb.adapters.datatypes.db.mapped_value import ValueWrapperIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.basic.profile import TvbProfile
from tvb.core import utils
from tvb.core.entities.load import try_get_last_datatype
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.storage import dao
from tvb.core.services.exceptions import ImportException
from tvb.core.services.figure_service import FigureService
from tvb.core.services.import_service import ImportService
from tvb.core.services.project_service import ProjectService
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.core.services.figure_service_test import IMG_DATA

FILE_IS_NONE = "Exported file is none"


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
        StorageInterface.remove_folder(TvbProfile.current.TVB_TEMP_FOLDER)

        # Delete folder where data was exported
        if self.zip_path:
            StorageInterface.remove_folder(os.path.split(self.zip_path)[0])

        self.delete_project_folders()


    def test_import_export(self, user_factory, project_factory, value_wrapper_factory):
        """
        Test the import/export mechanism for a project structure.
        The project contains the following data types: Connectivity, Surface, MappedArray and ValueWrapper.
        """
        test_user = user_factory()
        test_project = project_factory(test_user, "TestImportExport", "test_desc")
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_66.zip')
        TestFactory.import_zip_connectivity(test_user, test_project, zip_path)
        value_wrapper = value_wrapper_factory(test_user, test_project)
        ProjectService.set_datatype_visibility(value_wrapper.gid, False)

        result = self.get_all_datatypes()
        expected_results = {}
        for one_data in result:
            expected_results[one_data.gid] = (one_data.module, one_data.type)

        # Export project as ZIP
        self.zip_path = ExportManager().export_project(test_project)
        assert self.zip_path is not None, FILE_IS_NONE

        # Remove the original project
        self.project_service.remove_project(test_project.id)
        result, lng_ = self.project_service.retrieve_projects_for_user(test_user.id)
        assert 0 == len(result), "Project Not removed!"
        assert 0 == lng_, "Project Not removed!"

        # Now try to import again project
        self.import_service.import_project_structure(self.zip_path, test_user.id)
        result = self.project_service.retrieve_projects_for_user(test_user.id)[0]
        assert len(result) == 1, "There should be only one project."
        assert result[0].name == "TestImportExport", "The project name is not correct."
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
        assert False == new_val.visible, "Visibility incorrectly restored"

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
        assert self.zip_path is not None, FILE_IS_NONE

        with pytest.raises(ImportException):
            self.import_service.import_project_structure(self.zip_path, test_user.id)

    def test_export_import_burst(self, user_factory, project_factory, simulation_launch):
        """
        Test that fk_parent_burst is correctly preserved after export/import
        """
        test_user = user_factory()
        test_project = project_factory(test_user, "TestIESim")
        sim_op = simulation_launch(test_user, test_project, simulation_length=10)
        tries = 5
        while not sim_op.has_finished and tries > 0:
            sleep(5)
            tries = tries - 1
            sim_op = dao.get_operation_by_id(sim_op.id)
        assert sim_op.has_finished, "Simulation did not finish in the given time"

        self.zip_path = ExportManager().export_project(test_project)
        assert self.zip_path is not None, FILE_IS_NONE
        self.project_service.remove_project(test_project.id)

        self.import_service.import_project_structure(self.zip_path, test_user.id)
        retrieved_project = self.project_service.retrieve_projects_for_user(test_user.id)[0][0]
        ts = try_get_last_datatype(retrieved_project.id, TimeSeriesRegionIndex)
        bursts = dao.get_bursts_for_project(retrieved_project.id)
        assert 1 == len(bursts)
        assert ts.fk_parent_burst == bursts[0].gid

    def test_export_import_figures(self, user_factory, project_factory):
        """
        Test that ResultFigure instances are correctly restores after an export+import project
        """
        # Prepare data
        user = user_factory()
        project = project_factory(user, "TestImportExportFigures")
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'paupau.zip')
        TestFactory.import_zip_connectivity(user, project, zip_path)

        figure_service = FigureService()
        figure_service.store_result_figure(project, user, "png", IMG_DATA, "bla")
        figure_service.store_result_figure(project, user, "png", IMG_DATA, "bla")
        figures = list(figure_service.retrieve_result_figures(project, user)[0].values())[0]
        assert 2 == len(figures)

        # export, delete and the import project
        self.zip_path = ExportManager().export_project(project)
        assert self.zip_path is not None, FILE_IS_NONE
        self.project_service.remove_project(project.id)

        self.import_service.import_project_structure(self.zip_path, user.id)

        # Check that state is as before export: one operation, one DT, 2 figures
        retrieved_project = self.project_service.retrieve_projects_for_user(user.id)[0][0]
        count_operations = dao.get_filtered_operations(retrieved_project.id, None, is_count=True)
        assert 1 == count_operations
        count_datatypes = dao.count_datatypes(retrieved_project.id, DataType)
        assert 1 == count_datatypes

        figures = list(figure_service.retrieve_result_figures(retrieved_project, user)[0].values())[0]
        assert 2 == len(figures)
        assert "bla" in figures[0].name
        assert "bla" in figures[1].name
        image_path = utils.url2path(figures[0].file_path)
        img_data = Image.open(image_path).load()
        assert img_data is not None
