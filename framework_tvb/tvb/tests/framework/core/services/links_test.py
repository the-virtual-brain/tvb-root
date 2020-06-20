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
Testing linking datatypes between projects.

.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import pytest
import os
import tvb_data
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.core.entities.file.files_helper import TvbZip, FilesHelper
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.import_service import ImportService
from tvb.datatypes.sensors import EEG_POLYMORPHIC_IDENTITY
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class _BaseLinksTest(TransactionalTestCase):

    @pytest.fixture()
    def initialize_two_projects(self, dummy_datatype_index_factory, project_factory, user_factory):
        """
        Creates a user, an algorithm and 2 projects
        Project src_project will have an operation and 2 datatypes
        Project dest_project will be empty.
        Initializes a flow and a project service
        """
        self.clean_database(delete_folders=True)

        self.algorithm_service = AlgorithmService()
        self.project_service = ProjectService()

        # Create the source project with 2 datatypes in it
        src_user = user_factory(username="Links Test")
        self.src_usr_id = src_user.id
        self.src_project = project_factory(src_user, "Src_Project")
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'paupau.zip')
        self.red_datatype = TestFactory.import_zip_connectivity(src_user, self.src_project, zip_path, "John")
        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'sensors', 'eeg_unitvector_62.txt.bz2')
        self.blue_datatype = TestFactory.import_sensors(src_user, self.src_project, zip_path, EEG_POLYMORPHIC_IDENTITY)
        assert 1 == self.red_datatypes_in(self.src_project.id)
        assert 1 == self.blue_datatypes_in(self.src_project.id)

        # create the destination project empty
        self.dst_user = user_factory(username='Destination_User')
        self.dst_usr_id = self.dst_user.id
        self.dest_project = project_factory(self.dst_user, "Destination_Project")
        assert 0 == self.red_datatypes_in(self.dest_project.id)
        assert 0 == self.blue_datatypes_in(self.dest_project.id)

    def transactional_teardown_method(self):
        self.clean_database(delete_folders=True)

    def red_datatypes_in(self, project_id):
        return dao.count_datatypes(project_id, ConnectivityIndex)

    def blue_datatypes_in(self, project_id):
        return dao.count_datatypes(project_id, SensorsIndex)


class TestLinks(_BaseLinksTest):
    """
    Test case for datatype linking functionality
    """

    def test_create_link(self, initialize_two_projects):
        dest_id = self.dest_project.id
        self.algorithm_service.create_link([self.red_datatype.id], dest_id)
        assert 1 == self.red_datatypes_in(dest_id)
        assert 0 == self.blue_datatypes_in(dest_id)

    def test_remove_link(self, initialize_two_projects):
        dest_id = self.dest_project.id
        self.algorithm_service.create_link([self.red_datatype.id], dest_id)
        assert 1 == self.red_datatypes_in(dest_id)
        self.algorithm_service.remove_link(self.red_datatype.id, dest_id)
        assert 0 == self.red_datatypes_in(dest_id)

    def test_link_appears_in_project_structure(self, initialize_two_projects):
        dest_id = self.dest_project.id
        self.algorithm_service.create_link([self.red_datatype.id], dest_id)
        # Test getting information about linked datatypes, from low level methods to the one used by the UI
        dt_1s = dao.get_linked_datatypes_in_project(dest_id)
        assert 1 == len(dt_1s)
        assert 1 == self.red_datatypes_in(dest_id)
        json = self.project_service.get_project_structure(self.dest_project, None, DataTypeMetaData.KEY_STATE,
                                                          DataTypeMetaData.KEY_SUBJECT, None)
        assert self.red_datatype.gid in json

    def test_remove_entity_with_links_moves_links(self, initialize_two_projects):
        project_path = FilesHelper().get_project_folder(self.src_project)
        self.red_datatype.storage_path = project_path
        dest_id = self.dest_project.id
        self.algorithm_service.create_link([self.red_datatype.id], dest_id)
        assert 1 == self.red_datatypes_in(dest_id)
        # remove original datatype
        self.project_service.remove_datatype(self.src_project.id, self.red_datatype.gid)
        # datatype has been moved to one of it's links
        assert 1 == self.red_datatypes_in(dest_id)
        # project dest no longer has a link but owns the data type
        dt_links = dao.get_linked_datatypes_in_project(dest_id)
        assert 0 == len(dt_links)


class TestImportExportProjectWithLinksTest(_BaseLinksTest):

    @pytest.fixture()
    def initialize_linked_projects(self, initialize_two_projects):
        """
        Adds to the _BaseLinksTest setup the following
        2 links from src to dest project
        Import/export services
        """

        dest_id = self.dest_project.id
        self.algorithm_service.create_link([self.red_datatype.id], dest_id)
        self.algorithm_service.create_link([self.blue_datatype.id], dest_id)
        self.export_mng = ExportManager()

    def test_export(self, initialize_linked_projects):
        export_file = self.export_mng.export_project(self.dest_project)
        with TvbZip(export_file) as z:
            assert 'links-to-external-projects/Operation.xml' in z.namelist()

    def _export_and_remove(self, project):
        """export the project and remove it"""
        export_file = self.export_mng.export_project(project)
        self.project_service.remove_project(project.id)
        return export_file

    def _import(self, export_file, user_id):
        """ import a project zip for a user """
        # instantiated for every use because it is stateful
        import_service = ImportService()
        import_service.import_project_structure(export_file, user_id)
        return import_service.created_projects[0].id

    def test_links_recreated_on_import(self, initialize_linked_projects):
        export_file = self._export_and_remove(self.dest_project)
        imported_proj_id = self._import(export_file, self.dst_usr_id)
        assert 1 == self.red_datatypes_in(imported_proj_id)
        assert 1 == self.blue_datatypes_in(imported_proj_id)
        links = dao.get_linked_datatypes_in_project(imported_proj_id)
        assert 2 == len(links)

    def test_datatypes_recreated_on_import(self, initialize_linked_projects):
        export_file = self._export_and_remove(self.dest_project)
        self.project_service.remove_project(self.src_project.id)
        # both projects have been deleted
        # import should recreate links as datatypes
        imported_proj_id = self._import(export_file, self.dst_usr_id)
        assert 1 == self.red_datatypes_in(imported_proj_id)
        assert 1 == self.blue_datatypes_in(imported_proj_id)
        links = dao.get_linked_datatypes_in_project(imported_proj_id)
        assert 0 == len(links)

    def test_datatypes_and_links_recreated_on_import(self, initialize_linked_projects):
        export_file = self._export_and_remove(self.dest_project)
        # remove datatype 2 from source project
        self.project_service.remove_datatype(self.src_project.id, self.blue_datatype.gid)
        imported_proj_id = self._import(export_file, self.dst_usr_id)
        # both datatypes should be recreated
        assert 1 == self.red_datatypes_in(imported_proj_id)
        assert 1 == self.blue_datatypes_in(imported_proj_id)
        # only datatype 1 should be a link
        links = dao.get_linked_datatypes_in_project(imported_proj_id)
        assert 1 == len(links)
        assert self.red_datatype.gid == links[0].gid

    @pytest.fixture()
    def create_interlinked_projects(self):
        def build():
            """
            Project src will have 3 datatypes, and a link to the VW from the dest project.
            Project dest will have the derived VW and links
            """
            # add a connectivity to src project and link it to dest project
            zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
            conn = TestFactory.import_zip_connectivity(self.dst_user, self.src_project, zip_path, "John")
            self.algorithm_service.create_link([conn.id], self.dest_project.id)

            # in dest derive a ValueWrapper from the linked conn
            vw_gid = TestFactory.create_value_wrapper(self.dst_user, self.dest_project)[1]
            vw = dao.get_datatype_by_gid(vw_gid)
            # then link the time series in the src project
            self.algorithm_service.create_link([vw.id], self.src_project.id)

            assert 3 == len(dao.get_datatypes_in_project(self.src_project.id))
            assert 1 == len(dao.get_linked_datatypes_in_project(self.src_project.id))
            assert 1 == len(dao.get_datatypes_in_project(self.dest_project.id))
            assert 3 == len(dao.get_linked_datatypes_in_project(self.dest_project.id))

        return build

    def test_create_interlinked_projects(self, initialize_linked_projects, create_interlinked_projects):
        create_interlinked_projects()

    def test_linked_datatype_dependencies_restored_on_import(self, initialize_linked_projects,
                                                             create_interlinked_projects):
        create_interlinked_projects()
        # export both then remove them
        export_file_src = self._export_and_remove(self.src_project)
        assert 4 == len(dao.get_datatypes_in_project(self.dest_project.id))
        assert 0 == len(dao.get_linked_datatypes_in_project(self.dest_project.id))
        export_file_dest = self._export_and_remove(self.dest_project)

        # importing both projects should work
        imported_id_1 = self._import(export_file_src, self.src_usr_id)
        assert 4 == len(dao.get_datatypes_in_project(imported_id_1))
        assert 0 == len(dao.get_linked_datatypes_in_project(imported_id_1))

        imported_id_2 = self._import(export_file_dest, self.dst_usr_id)
        assert 0 == len(dao.get_datatypes_in_project(imported_id_2))
        assert 4 == len(dao.get_linked_datatypes_in_project(imported_id_2))

    def test_linked_datatype_dependencies_restored_on_import_inverse_order(self, initialize_linked_projects,
                                                                           create_interlinked_projects):
        create_interlinked_projects()
        # export both then remove them
        export_file_src = self._export_and_remove(self.src_project)
        assert 4 == len(dao.get_datatypes_in_project(self.dest_project.id))
        assert 0 == len(dao.get_linked_datatypes_in_project(self.dest_project.id))
        export_file_dest = self._export_and_remove(self.dest_project)

        # importing dest before src should work
        imported_id_2 = self._import(export_file_dest, self.dst_usr_id)
        assert 4 == len(dao.get_datatypes_in_project(imported_id_2))
        assert 0 == len(dao.get_linked_datatypes_in_project(imported_id_2))

        imported_id_1 = self._import(export_file_src, self.src_usr_id)
        assert 0 == len(dao.get_datatypes_in_project(imported_id_1))
        assert 4 == len(dao.get_linked_datatypes_in_project(imported_id_1))
