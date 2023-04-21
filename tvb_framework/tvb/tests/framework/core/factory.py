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

 A Factory class to be used in tests for generating default entities:
Project, User, Operation, basic imports (e.g. CFF).

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import os
import random
import uuid
import tvb_data

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.h5.mapped_value_h5 import ValueWrapper
from tvb.adapters.uploaders.gifti_surface_importer import GIFTISurfaceImporter, GIFTISurfaceImporterModel
from tvb.adapters.uploaders.obj_importer import ObjSurfaceImporter, ObjSurfaceImporterModel
from tvb.adapters.uploaders.projection_matrix_importer import ProjectionMatrixImporterModel
from tvb.adapters.uploaders.projection_matrix_importer import ProjectionMatrixSurfaceEEGImporter
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporterModel, RegionMappingImporter
from tvb.adapters.uploaders.sensors_importer import SensorsImporterModel, SensorsImporter
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel, ZIPConnectivityImporter
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporter, ZIPSurfaceImporterModel
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.load import try_get_last_datatype
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.model.model_operation import *
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.neotraits.view_model import ViewModel
from tvb.core.services.burst_service import BurstService
from tvb.core.services.import_service import ImportService
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import ProjectService
from tvb.core.utils import hash_password
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.surfaces import CorticalSurface
from tvb.storage.storage_interface import StorageInterface


class TestFactory(object):
    """
    Expose mostly static methods for creating different entities used in tests.
    """

    @staticmethod
    def get_entity(project, expected_data, filters=None):
        """
        Return the first entity with class given by `expected_data`

        :param expected_data: specifies the class whose entity is returned
        """
        return try_get_last_datatype(project.id, expected_data, filters)

    @staticmethod
    def get_entity_count(project, datatype_class):
        """
        Return the count of stored datatypes with class given by `datatype`

        :param datatype: Take class from this instance amd count for this class
        """
        return dao.count_datatypes(project.id, datatype_class)

    @staticmethod
    def _assert_one_more_datatype(project, dt_class, prev_count=0):
        dt = try_get_last_datatype(project.id, dt_class)
        count = dao.count_datatypes(project.id, dt_class)
        assert prev_count + 1 == count, "Project should contain only one new DT."
        assert dt is not None, "Retrieved DT should not be empty"
        return dt

    @staticmethod
    def create_user(username='test_user_42', display_name='test_display_name', password='test_pass',
                    mail='test_mail@tvb.org', validated=True, role='test'):
        """
        Create persisted User entity.

        :returns: User entity after persistence.
        """
        user = User(username, display_name, password, mail, validated, role)
        return dao.store_entity(user)

    @staticmethod
    def create_project(admin, name="TestProject42", description='description', users=None):
        """
        Create persisted Project entity, with no linked DataTypes.

        :returns: Project entity after persistence.
        """
        if users is None:
            users = []
        data = dict(name=name, description=description, users=users, max_operation_size=None, disable_imports=False)
        return ProjectService().store_project(admin, True, None, **data)

    @staticmethod
    def create_figure(user_id, project_id, session_name=None, name=None, path=None, file_format='PNG'):
        """
        :returns: the `ResultFigure` for a result with the given specifications
        """
        figure = ResultFigure(user_id, project_id, session_name, name, path, file_format)
        return dao.store_entity(figure)

    @staticmethod
    def create_operation(test_user=None, test_project=None, operation_status=STATUS_FINISHED):
        """
        Create persisted operation.
        :return: Operation entity after persistence.
        """
        if test_user is None:
            test_user = TestFactory.create_user()
        if test_project is None:
            test_project = TestFactory.create_project(test_user)

        algorithm = dao.get_algorithm_by_module(TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS)
        adapter = ABCAdapter.build_adapter(algorithm)
        view_model = adapter.get_view_model_class()()
        view_model.data_file = "."
        operation = Operation(view_model.gid.hex, test_user.id, test_project.id, algorithm.id,
                              status=operation_status)
        dao.store_entity(operation)
        op_dir = StorageInterface().get_project_folder(test_project.name, str(operation.id))
        h5.store_view_model(view_model, op_dir)
        return dao.get_operation_by_id(operation.id)

    @staticmethod
    def create_value_wrapper(test_user, test_project=None):
        """
        Creates a ValueWrapper dataType, and the associated parent Operation.
        This is also used in ProjectStructureTest.
        """
        if test_project is None:
            test_project = TestFactory.create_project(test_user, 'test_proj')
        operation = TestFactory.create_operation(test_user=test_user, test_project=test_project)
        value_wrapper = ValueWrapper(data_value="5.0", data_name="my_value", data_type="float")
        vw_idx = h5.store_complete(value_wrapper, operation.id, operation.project.name)
        vw_idx.fk_from_operation = operation.id
        vw_idx = dao.store_entity(vw_idx)
        return test_project, vw_idx.gid, operation

    @staticmethod
    def create_local_connectivity(user, project, surface_gid):

        op = TestFactory.create_operation(test_user=user, test_project=project)

        wrapper_surf = CorticalSurface()
        wrapper_surf.gid = uuid.UUID(surface_gid)
        lc_ht = LocalConnectivity.from_file()
        lc_ht.surface = wrapper_surf
        lc_idx = h5.store_complete(lc_ht, op.id, project.name)
        lc_idx.fk_surface_gid = surface_gid
        lc_idx.fk_from_operation = op.id
        dao.store_entity(lc_idx)

        return TestFactory._assert_one_more_datatype(project, LocalConnectivityIndex)

    @staticmethod
    def create_adapter(module='tvb.tests.framework.adapters.ndimensionarrayadapter',
                       class_name='NDimensionArrayAdapter'):
        """
        :returns: Adapter Class after initialization.
        """
        algorithm = dao.get_algorithm_by_module(module, class_name)
        return ABCAdapter.build_adapter(algorithm)

    @staticmethod
    def store_burst(project_id, operation=None):
        """
        Build and persist BurstConfiguration entity.
        """
        burst = BurstConfiguration(project_id)
        if operation is not None:
            burst.name = 'dummy_burst'
            burst.status = BurstConfiguration.BURST_FINISHED
            burst.start_time = datetime.now()
            burst.range1 = '["conduction_speed", {"lo": 50, "step": 1.0, "hi": 100.0}]'
            burst.range2 = '["connectivity", null]'
            burst.fk_simulation = operation.id
            burst.simulator_gid = uuid.uuid4().hex
            BurstService().store_burst_configuration(burst)
        return dao.store_entity(burst)

    @staticmethod
    def import_default_project(admin_user=None):

        if not admin_user:
            admin_user = TestFactory.create_user()

        project_path = os.path.join(os.path.dirname(tvb_data.__file__), 'Default_Project.zip')
        import_service = ImportService()
        import_service.import_project_structure(project_path, admin_user.id)
        return import_service.created_projects[0]

    @staticmethod
    def launch_importer(importer_class, view_model, user, project, same_process=True):
        # type: (type, ViewModel, User, Project, bool) -> None
        """
        same_process = False will do the normal flow, with Uploaders running synchronously but in a different process.
        This branch won't be compatible with usage in subclasses of TransactionalTestCase because the upload results
        won't be available for the unit-test running.
        same_process = True for usage in subclasses of TransactionalTestCase, as data preparation, for example. Won't
        test the "real" upload flow, but it is very close to that.
        """
        importer = ABCAdapter.build_adapter_from_class(importer_class)
        if same_process:
            TestFactory.launch_synchronously(user.id, project, importer, view_model)
        else:
            OperationService().fire_operation(importer, user, project.id, view_model=view_model)

    @staticmethod
    def import_region_mapping(user, project, import_file_path, surface_gid, connectivity_gid, same_process=True):

        view_model = RegionMappingImporterModel()
        view_model.mapping_file = import_file_path
        view_model.surface = surface_gid
        view_model.connectivity = connectivity_gid
        TestFactory.launch_importer(RegionMappingImporter, view_model, user, project, same_process)

        return TestFactory._assert_one_more_datatype(project, RegionMappingIndex)

    @staticmethod
    def import_surface_gifti(user, project, path, same_process=False):

        view_model = GIFTISurfaceImporterModel()
        view_model.data_file = path
        view_model.should_center = False
        TestFactory.launch_importer(GIFTISurfaceImporter, view_model, user, project, same_process)

        return TestFactory._assert_one_more_datatype(project, SurfaceIndex)

    @staticmethod
    def import_surface_zip(user, project, zip_path, surface_type, zero_based=True, same_process=True):

        count = dao.count_datatypes(project.id, SurfaceIndex)

        view_model = ZIPSurfaceImporterModel()
        view_model.uploaded = zip_path
        view_model.should_center = True
        view_model.zero_based_triangles = zero_based
        view_model.surface_type = surface_type
        TestFactory.launch_importer(ZIPSurfaceImporter, view_model, user, project, same_process)

        return TestFactory._assert_one_more_datatype(project, SurfaceIndex, count)

    @staticmethod
    def import_surface_obj(user, project, obj_path, surface_type, same_process=True):

        view_model = ObjSurfaceImporterModel()
        view_model.data_file = obj_path
        view_model.surface_type = surface_type
        TestFactory.launch_importer(ObjSurfaceImporter, view_model, user, project, same_process)

        return TestFactory._assert_one_more_datatype(project, SurfaceIndex)

    @staticmethod
    def import_sensors(user, project, zip_path, sensors_type, same_process=True):

        view_model = SensorsImporterModel()
        view_model.sensors_file = zip_path
        view_model.sensors_type = sensors_type
        TestFactory.launch_importer(SensorsImporter, view_model, user, project, same_process)

        return TestFactory._assert_one_more_datatype(project, SensorsIndex)

    @staticmethod
    def import_projection_matrix(user, project, file_path, sensors_gid, surface_gid, same_process=True):

        view_model = ProjectionMatrixImporterModel()
        view_model.projection_file = file_path
        view_model.sensors = sensors_gid
        view_model.surface = surface_gid
        TestFactory.launch_importer(ProjectionMatrixSurfaceEEGImporter, view_model, user, project, same_process)

        return TestFactory._assert_one_more_datatype(project, ProjectionMatrixIndex)

    @staticmethod
    def import_zip_connectivity(user, project, zip_path=None, subject=DataTypeMetaData.DEFAULT_SUBJECT,
                                same_process=True):

        if zip_path is None:
            zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        count = dao.count_datatypes(project.id, ConnectivityIndex)

        view_model = ZIPConnectivityImporterModel()
        view_model.uploaded = zip_path
        view_model.data_subject = subject
        TestFactory.launch_importer(ZIPConnectivityImporter, view_model, user, project, same_process)

        return TestFactory._assert_one_more_datatype(project, ConnectivityIndex, count)

    @staticmethod
    def launch_synchronously(test_user_id, test_project, adapter_instance, view_model):
        # Avoid the scheduled execution, as this is asynch, thus launch it immediately
        service = OperationService()
        algorithm = adapter_instance.stored_adapter

        operation = service.prepare_operation(test_user_id, test_project, algorithm, True, view_model)
        service.initiate_prelaunch(operation, adapter_instance)

        operation = dao.get_operation_by_id(operation.id)
        # Check that operation status after execution is success.
        assert STATUS_FINISHED == operation.status
        # Make sure at least one result exists for each BCT algorithm
        return dao.get_generic_entity(DataType, operation.id, 'fk_from_operation')


class ExtremeTestFactory(object):
    """
    Test Factory for random and large number of users.
    """

    VALIDATION_DICT = {}

    @staticmethod
    def get_users_ids(wanted_nr, total_nr, exclude_id, available_users):
        """
        Generate random users
        """
        nr_users = 0
        result = []
        while nr_users < wanted_nr:
            new_idx = random.randint(0, total_nr - 1)
            new_id = available_users[new_idx].id
            if new_id not in result and new_id != exclude_id:
                ExtremeTestFactory.VALIDATION_DICT[new_id] += 1
                result.append(new_id)
                nr_users += 1
        return result

    @staticmethod
    def generate_users(nr_users, nr_projects):
        """
        The generate_users method will create a clean state db with
        :param nr_users: number of users to be generated (with random roles between
                                CLINICIAN and RESEARCHER and random validated state)
        :param nr_projects: maximum number of projects to be generated for each user
        """
        users = []

        for i in range(nr_users):
            coin_flip = random.randint(0, 1)
            role = 'CLINICIAN' if coin_flip == 1 else 'RESEARCHER'
            password = hash_password("test")
            new_user = User("gen" + str(i), "name" + str(i), password, "test_mail@tvb.org", True, role)
            dao.store_entity(new_user)
            new_user = dao.get_user_by_name("gen" + str(i))
            ExtremeTestFactory.VALIDATION_DICT[new_user.id] = 0
            users.append(new_user)

        for i in range(nr_users):
            current_user = dao.get_user_by_name("gen" + str(i))
            projects_for_user = random.randint(0, nr_projects)
            for j in range(projects_for_user):
                data = dict(name='GeneratedProject' + str(i) + '_' + str(j),
                            description='test_desc',
                            users=ExtremeTestFactory.get_users_ids(random.randint(0, nr_users - 3),
                                                                   nr_users, current_user.id, users),
                            max_operation_size=None,
                            disable_imports=False)
                ProjectService().store_project(current_user, True, None, **data)
                ExtremeTestFactory.VALIDATION_DICT[current_user.id] += 1
