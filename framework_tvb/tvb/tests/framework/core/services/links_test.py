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
Testing linking datatypes between projects.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.adapters.exporters.export_manager import ExportManager
from tvb.core.entities.file.files_helper import TvbZip
from tvb.core.entities.storage import dao
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.import_service import ImportService
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.datatypes.datatype2 import Datatype2


class _BaseLinksTest(TransactionalTestCase):

    GEORGE1st = "george the grey"
    GEORGE2nd = "george"


    def _initialize_two_projects(self):
        """
        Creates a user, an algorithm and 2 projects
        Project src_project will have an operation and 2 datatypes
        Project dest_project will be empty.
        Initializes a flow and a project service
        """
        self.datatype_factory_src = DatatypesFactory()
        self.src_project = self.datatype_factory_src.project
        self.src_usr_id = self.datatype_factory_src.user.id

        self.red_datatype = self.datatype_factory_src.create_simple_datatype(subject=self.GEORGE1st)
        self.blue_datatype = self.datatype_factory_src.create_datatype_with_storage(subject=self.GEORGE2nd)

        # create the destination project
        self.datatype_factory_dest = DatatypesFactory()
        self.dest_project = self.datatype_factory_dest.project
        self.dest_usr_id = self.datatype_factory_dest.user.id

        self.flow_service = FlowService()
        self.project_service = ProjectService()


    def transactional_setup_method(self):
        self.clean_database(delete_folders=True)
        self._initialize_two_projects()


    def transactional_teardown_method(self):
        self.clean_database(delete_folders=True)


    def red_datatypes_in(self, project_id):
        return self.flow_service.get_available_datatypes(project_id, Datatype1)[1]


    def blue_datatypes_in(self, project_id):
        return self.flow_service.get_available_datatypes(project_id, Datatype2)[1]



class TestLinks(_BaseLinksTest):
    """
    Test case for datatype linking functionality
    """

    def assertRedsInDest(self, count):
        assert count == self.red_datatypes_in(self.dest_project.id)


    def test_create_link(self):
        dest_id = self.dest_project.id
        assert 0 == self.red_datatypes_in(dest_id)
        self.flow_service.create_link([self.red_datatype.id], dest_id)
        assert 1 == self.red_datatypes_in(dest_id)
        assert 0 == self.blue_datatypes_in(dest_id)

    def test_remove_link(self):
        dest_id = self.dest_project.id
        assert 0 == self.red_datatypes_in(dest_id)
        self.flow_service.create_link([self.red_datatype.id], dest_id)
        assert 1 == self.red_datatypes_in(dest_id)
        self.flow_service.remove_link(self.red_datatype.id, dest_id)
        assert 0 == self.red_datatypes_in(dest_id)


    def test_link_appears_in_project_structure(self):
        dest_id = self.dest_project.id
        self.flow_service.create_link([self.red_datatype.id], dest_id)
        # Test getting information about linked datatypes, from low level methods to the one used by the UI
        dt_1s = dao.get_linked_datatypes_in_project(dest_id)
        assert 1 == len(dt_1s)
        assert 1 == self.red_datatypes_in(dest_id)
        json = self.project_service.get_project_structure(self.dest_project, None, DataTypeMetaData.KEY_STATE,
                                                          DataTypeMetaData.KEY_SUBJECT, None)
        assert self.red_datatype.gid in json


    def test_remove_entity_with_links_moves_links(self):
        dest_id = self.dest_project.id
        self.flow_service.create_link([self.red_datatype.id], dest_id)
        assert 1 == self.red_datatypes_in(dest_id)
        # remove original datatype
        self.project_service.remove_datatype(self.src_project.id, self.red_datatype.gid)
        # datatype has been moved to one of it's links
        assert 1 == self.red_datatypes_in(dest_id)
        # project dest no longer has a link but owns the data type
        dt_links = dao.get_linked_datatypes_in_project(dest_id)
        assert 0 == len(dt_links)


class ImportExportProjectWithLinksTest(_BaseLinksTest):

    def transactional_setup_method(self):
        """
        Adds to the _BaseLinksTest setup the following
        2 links from src to dest project
        Import/export services
        """
        self.clean_database(delete_folders=True)
        self._initialize_two_projects()

        dest_id = self.dest_project.id
        self.flow_service.create_link([self.red_datatype.id], dest_id)
        self.flow_service.create_link([self.blue_datatype.id], dest_id)
        self.export_mng = ExportManager()


    def test_export(self):
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


    def test_links_recreated_on_import(self):
        export_file = self._export_and_remove(self.dest_project)
        imported_proj_id = self._import(export_file, self.dest_usr_id)
        assert 1 == self.red_datatypes_in(imported_proj_id)
        assert 1 == self.blue_datatypes_in(imported_proj_id)
        links = dao.get_linked_datatypes_in_project(imported_proj_id)
        assert 2 == len(links)


    def test_datatypes_recreated_on_import(self):
        export_file = self._export_and_remove(self.dest_project)
        self.project_service.remove_project(self.src_project.id)
        # both projects have been deleted
        # import should recreate links as datatypes
        imported_proj_id = self._import(export_file, self.dest_usr_id)
        assert 1 == self.red_datatypes_in(imported_proj_id)
        assert 1 == self.blue_datatypes_in(imported_proj_id)
        links = dao.get_linked_datatypes_in_project(imported_proj_id)
        assert 0 == len(links)


    def test_datatypes_and_links_recreated_on_import(self):
        export_file = self._export_and_remove(self.dest_project)
        # remove datatype 2 from source project
        self.project_service.remove_datatype(self.src_project.id, self.blue_datatype.gid)
        imported_proj_id = self._import(export_file, self.dest_usr_id)
        # both datatypes should be recreated
        assert 1 == self.red_datatypes_in(imported_proj_id)
        assert 1 == self.blue_datatypes_in(imported_proj_id)
        # only datatype 1 should be a link
        links = dao.get_linked_datatypes_in_project(imported_proj_id)
        assert 1 == len(links)
        assert self.red_datatype.gid == links[0].gid


    def _create_interlinked_projects(self):
        """
        Extend the two projects created in setup.
        Project src will have 3 datatypes, one a connectivity, and a link to the time series from the dest project.
        Project dest will have 3 links to the datatypes in src and a time series derived from the linked connectivity
        """
        # add a connectivity to src project and link it to dest project
        _, conn = self.datatype_factory_src.create_connectivity()
        self.flow_service.create_link([conn.id], self.dest_project.id)
        # in dest derive a time series from the linked conn
        ts = self.datatype_factory_dest.create_timeseries(conn)
        # then link the time series in the src project
        self.flow_service.create_link([ts.id], self.src_project.id)

        assert 3 == len(dao.get_datatypes_in_project(self.src_project.id))
        assert 1 == len(dao.get_linked_datatypes_in_project(self.src_project.id))
        assert 1 == len(dao.get_datatypes_in_project(self.dest_project.id))
        assert 3 == len(dao.get_linked_datatypes_in_project(self.dest_project.id))


    def test_create_interlinked_projects(self):
        self._create_interlinked_projects()


    def test_linked_datatype_dependencies_restored_on_import(self):
        self._create_interlinked_projects()
        # export both then remove them
        export_file_src = self._export_and_remove(self.src_project)
        assert 4 == len(dao.get_datatypes_in_project(self.dest_project.id))
        assert 0 == len(dao.get_linked_datatypes_in_project(self.dest_project.id))
        export_file_dest = self._export_and_remove(self.dest_project)

        # importing both projects should work
        imported_id_1 = self._import(export_file_src, self.src_usr_id)
        assert 4 == len(dao.get_datatypes_in_project(imported_id_1))
        assert 0 == len(dao.get_linked_datatypes_in_project(imported_id_1))

        imported_id_2 = self._import(export_file_dest, self.dest_usr_id)
        assert 0 == len(dao.get_datatypes_in_project(imported_id_2))
        assert 4 == len(dao.get_linked_datatypes_in_project(imported_id_2))


    def test_linked_datatype_dependencies_restored_on_import_inverse_order(self):
        self._create_interlinked_projects()
        # export both then remove them
        export_file_src = self._export_and_remove(self.src_project)
        assert 4 == len(dao.get_datatypes_in_project(self.dest_project.id))
        assert 0 == len(dao.get_linked_datatypes_in_project(self.dest_project.id))
        export_file_dest = self._export_and_remove(self.dest_project)

        # importing dest before src should work
        imported_id_2 = self._import(export_file_dest, self.dest_usr_id)
        assert 4 == len(dao.get_datatypes_in_project(imported_id_2))
        assert 0 == len(dao.get_linked_datatypes_in_project(imported_id_2))

        imported_id_1 = self._import(export_file_src, self.src_usr_id)
        assert 0 == len(dao.get_datatypes_in_project(imported_id_1))
        assert 4 == len(dao.get_linked_datatypes_in_project(imported_id_1))

