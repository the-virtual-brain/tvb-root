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
.. moduleauthor:: Gabriel Florea <gabriel.florea@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import shutil
import pytest

from tvb.adapters.exporters.export_manager import ExportManager
from tvb.adapters.uploaders.tvb_importer import TVBImporterModel, TVBImporter
from tvb.basic.profile import TvbProfile
from tvb.core.entities.load import get_filtered_datatypes, load_entity_by_gid
from tvb.core.entities.storage import dao
from tvb.core.services.exceptions import OperationException
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestTVBImporter(BaseTestCase):
    """
    Unit-tests for TVB importer.
    """
    TVB_EXPORTER = "TVBExporter"
    TVB_LINKED_EXPORTER = "TVBLinkedExporter"

    @pytest.fixture()
    def prepare_importer_data(self, user_factory, project_factory, operation_factory,
                              connectivity_index_factory, datatype_group_factory):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a datatype and a datatype_group;
        """
        self.test_user = user_factory()
        self.test_project = project_factory(self.test_user)
        operation = operation_factory(test_project=self.test_project)

        # Generate simple data type and export it to H5 file
        self.datatype = connectivity_index_factory(op=operation)

        export_manager = ExportManager()
        _, exported_h5_file, _ = export_manager.export_data(self.datatype, self.TVB_EXPORTER, self.test_project)

        # Copy H5 file to another location since the original one / exported will be deleted with the project
        _, h5_file_name = os.path.split(exported_h5_file)
        shutil.copy(exported_h5_file, TvbProfile.current.TVB_TEMP_FOLDER)
        self.h5_file_path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, h5_file_name)
        assert os.path.exists(self.h5_file_path), "Simple data type was not exported correct"

        # Generate data type group and export it to ZIP file
        datatype_group, _ = datatype_group_factory(project=self.test_project, store_vm=True)
        _, self.zip_file_path, _ = export_manager.export_data(datatype_group, self.TVB_EXPORTER, self.test_project)
        assert os.path.exists(self.zip_file_path), "Data type group was not exported correct"

        self.clean_database(delete_folders=False)

        # Recreate project, but a clean one where to import data
        self.test_user = user_factory()
        self.test_project = project_factory(self.test_user)

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()

    def _import(self, import_file_path=None):

        view_model = TVBImporterModel()
        view_model.data_file = import_file_path
        TestFactory.launch_importer(TVBImporter, view_model, self.test_user, self.test_project, False)

    def test_zip_import(self, prepare_importer_data):
        """
        This method tests import of TVB data in zip format (which imply multiple data types
        in the same zip file - exported from a group)
        """
        self._import(self.zip_file_path)
        _, count = get_filtered_datatypes(self.test_project.id, self.datatype.module + "." + self.datatype.type)
        assert 9, count == "9 datatypes should have been imported from group."

    def test_h5_import(self, prepare_importer_data):
        """
        This method tests import of TVB data in h5 format. Single data type / import
        """
        self._import(self.h5_file_path)

        data_types, count = get_filtered_datatypes(self.test_project.id,
                                                   self.datatype.module + "." + self.datatype.type)
        assert 1, len(data_types) == "Project should contain only one data type."
        assert 1, count == "Project should contain only one data type."

        data_type_entity = load_entity_by_gid(data_types[0][2])
        assert data_type_entity is not None, "Datatype should not be none"
        assert self.datatype.gid, data_type_entity.gid == "Imported datatype should have the same gid"

    def test_import_datatype_with_links(self, region_mapping_index_factory, user_factory, project_factory):
        """
        This is a test for importing region mapping with links, that results in importing:
        connectivity, surface and region mapping all from one zip.
        """
        self.test_user = user_factory()
        self.test_project = project_factory(self.test_user)

        region_mapping_index = region_mapping_index_factory()

        export_manager = ExportManager()
        _, exported_h5_file, _ = export_manager.export_data(region_mapping_index, self.TVB_LINKED_EXPORTER, self.test_project)

        # Clean DB
        self.clean_database(delete_folders=False)

        # Recreate project, but a clean one where to import data
        self.test_user = user_factory()
        self.test_project = project_factory(self.test_user)

        datatypes = dao.get_datatypes_in_project(self.test_project.id)
        assert 0 == len(datatypes), "There are no DT's in DB before import."

        self._import(exported_h5_file)

        datatypes = dao.get_datatypes_in_project(self.test_project.id)
        assert 3 == len(datatypes), "Project should contain 3 data types."

        has_conn = False
        has_surface = False
        for dt in datatypes:
            if dt.gid == region_mapping_index.fk_connectivity_gid:
                has_conn = True
            if dt.gid == region_mapping_index.fk_surface_gid:
                has_surface = True

        assert has_conn is True, "Connectivity was imported as linked"
        assert has_surface is True, "Surface was imported as linked"

    def test_import_invalid_file(self, prepare_importer_data):
        """
        This method tests import of a file which does not exists or does not
        have a supported format.
        """
        try:
            self._import("invalid_path")
            raise AssertionError("System should throw an exception if trying to import an invalid file")
        except OperationException:
            # Expected
            pass

        # Now try to generate a file on disk with wrong format and import that
        file_path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, "dummy_file.txt")
        with open(file_path, "w") as f:
            f.write("dummy text")

        try:
            self._import(file_path)
            raise AssertionError("System should throw an exception if trying to import a file with wrong format")
        except OperationException:
            # Expected
            pass
