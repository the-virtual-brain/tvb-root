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
from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.h5.mapped_value_h5 import ValueWrapper
from tvb.adapters.uploaders.projection_matrix_importer import ProjectionMatrixImporterModel
from tvb.adapters.uploaders.projection_matrix_importer import ProjectionMatrixSurfaceEEGImporter
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporterModel, RegionMappingImporter
from tvb.adapters.uploaders.gifti_surface_importer import GIFTISurfaceImporter, GIFTISurfaceImporterModel
from tvb.adapters.uploaders.obj_importer import ObjSurfaceImporter, ObjSurfaceImporterModel
from tvb.adapters.uploaders.sensors_importer import SensorsImporterModel, SensorsImporter
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel, ZIPConnectivityImporter
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporter, ZIPSurfaceImporterModel
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.load import try_get_last_datatype
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neocom import h5
from tvb.core.services.burst_service import BurstService
from tvb.core.neotraits.view_model import ViewModel
from tvb.core.utils import hash_password
from tvb.core.entities.model.model_operation import *
from tvb.core.entities.storage import dao
from tvb.core.entities.model.model_burst import RANGE_PARAMETER_1
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.project_service import ProjectService
from tvb.core.services.import_service import ImportService
from tvb.core.services.operation_service import OperationService
from tvb.core.adapters.abcadapter import ABCAdapter


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
        data = dict(name=name, description=description, users=users)
        return ProjectService().store_project(admin, True, None, **data)

    @staticmethod
    def create_figure(operation_id, user_id, project_id, session_name=None,
                      name=None, path=None, file_format='PNG'):
        """
        :returns: the `ResultFigure` for a result with the given specifications
        """
        figure = ResultFigure(operation_id, user_id, project_id,
                              session_name, name, path, file_format)
        return dao.store_entity(figure)

    @staticmethod
    def create_operation(algorithm=None, test_user=None, test_project=None,
                         operation_status=STATUS_FINISHED, parameters="test params"):
        """
        Create persisted operation.

        :param algorithm: When not None, introspect TVB and TVB_TEST for adapters.
        :return: Operation entity after persistence.
        """
        if algorithm is None:
            algorithm = dao.get_algorithm_by_module('tvb.adapters.simulator.simulator_adapter', 'SimulatorAdapter')

        if test_user is None:
            test_user = TestFactory.create_user()

        if test_project is None:
            test_project = TestFactory.create_project(test_user)

        meta = {DataTypeMetaData.KEY_SUBJECT: "John Doe",
                DataTypeMetaData.KEY_STATE: "RAW_DATA"}
        operation = Operation(test_user.id, test_project.id, algorithm.id, parameters, meta=json.dumps(meta),
                              status=operation_status)
        dao.store_entity(operation)
        # Make sure lazy attributes are correctly loaded.
        return dao.get_operation_by_id(operation.id)

    @staticmethod
    def create_group(test_user=None, test_project=None, subject="John Doe"):
        """
        Create a group of 2 operations, each with at least one resultant DataType.
        """
        if test_user is None:
            test_user = TestFactory.create_user()
        if test_project is None:
            test_project = TestFactory.create_project(test_user)

        adapter_inst = TestFactory.create_adapter('tvb.tests.framework.adapters.testadapter3', 'TestAdapter3')
        adapter_inst.meta_data = {DataTypeMetaData.KEY_SUBJECT: subject}
        args = {RANGE_PARAMETER_1: 'param_5', 'param_5': [1, 2]}
        algo = adapter_inst.stored_adapter
        algo_category = dao.get_category_by_id(algo.fk_category)

        # Prepare Operations group. Execute them synchronously
        service = OperationService()
        operations = service.prepare_operations(test_user.id, test_project, algo, algo_category, {}, **args)[0]
        service.launch_operation(operations[0].id, False, adapter_inst)
        service.launch_operation(operations[1].id, False, adapter_inst)

        resulted_dts = dao.get_datatype_in_group(operation_group_id=operations[0].fk_operation_group)
        return resulted_dts, operations[0].fk_operation_group

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
        op_dir = FilesHelper().get_project_folder(test_project, str(operation.id))
        vw_idx = h5.store_complete(value_wrapper, op_dir)
        vw_idx.fk_from_operation = operation.id
        vw_idx = dao.store_entity(vw_idx)
        return test_project, vw_idx.gid, operation

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
            burst.start_time = datetime.datetime.now()
            burst.range1 = '["conduction_speed", {"lo": 50, "step": 1.0, "hi": 100.0}]'
            burst.range2 = '["connectivity", null]'
            burst.fk_simulation = operation.id
            burst.simulator_gid = uuid.uuid4().hex
            BurstService().update_burst_configuration_h5(burst)
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
    def launch_importer(importer_class, view_model, user, project_id):
        # type: (type, ViewModel, User, int) -> None
        importer = ABCAdapter.build_adapter_from_class(importer_class)
        OperationService().fire_operation(importer, user, project_id, view_model=view_model)

    @staticmethod
    def import_region_mapping(user, project, import_file_path, surface_gid, connectivity_gid):

        view_model = RegionMappingImporterModel()
        view_model.mapping_file = import_file_path
        view_model.surface = surface_gid
        view_model.connectivity = connectivity_gid
        TestFactory.launch_importer(RegionMappingImporter, view_model, user, project.id)

        return TestFactory._assert_one_more_datatype(project, RegionMappingIndex)

    @staticmethod
    def import_surface_gifti(user, project, path):

        view_model = GIFTISurfaceImporterModel()
        view_model.data_file = path
        view_model.should_center = False
        TestFactory.launch_importer(GIFTISurfaceImporter, view_model, user, project.id)

        return TestFactory._assert_one_more_datatype(project, SurfaceIndex)

    @staticmethod
    def import_surface_zip(user, project, zip_path, surface_type, zero_based=True):

        count = dao.count_datatypes(project.id, SurfaceIndex)

        view_model = ZIPSurfaceImporterModel()
        view_model.uploaded = zip_path
        view_model.should_center = True
        view_model.zero_based_triangles = zero_based
        view_model.surface_type = surface_type
        TestFactory.launch_importer(ZIPSurfaceImporter, view_model, user, project.id)

        return TestFactory._assert_one_more_datatype(project, SurfaceIndex, count)

    @staticmethod
    def import_surface_obj(user, project, obj_path, surface_type):

        view_model = ObjSurfaceImporterModel()
        view_model.data_file = obj_path
        view_model.surface_type = surface_type
        TestFactory.launch_importer(ObjSurfaceImporter, view_model, user, project.id)

        return TestFactory._assert_one_more_datatype(project, SurfaceIndex)

    @staticmethod
    def import_sensors(user, project, zip_path, sensors_type):

        view_model = SensorsImporterModel()
        view_model.sensors_file = zip_path
        view_model.sensors_type = sensors_type
        TestFactory.launch_importer(SensorsImporter, view_model, user, project.id)

        return TestFactory._assert_one_more_datatype(project, SensorsIndex)

    @staticmethod
    def import_projection_matrix(user, project, file_path, sensors_gid, surface_gid):

        view_model = ProjectionMatrixImporterModel()
        view_model.projection_file = file_path
        view_model.sensors = sensors_gid
        view_model.surface = surface_gid
        TestFactory.launch_importer(ProjectionMatrixSurfaceEEGImporter, view_model, user, project.id)

        return TestFactory._assert_one_more_datatype(project, ProjectionMatrixIndex)

    @staticmethod
    def import_zip_connectivity(user, project, zip_path=None, subject=DataTypeMetaData.DEFAULT_SUBJECT):

        if zip_path is None:
            zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        count = dao.count_datatypes(project.id, ConnectivityIndex)

        view_model = ZIPConnectivityImporterModel()
        view_model.uploaded = zip_path
        view_model.data_subject = subject
        TestFactory.launch_importer(ZIPConnectivityImporter, view_model, user, project.id)

        return TestFactory._assert_one_more_datatype(project, ConnectivityIndex, count)

    @staticmethod
    def launch_synchronously(test_user, test_project, adapter_instance, view_model, algo_category=None):
        # Avoid the scheduled execution, as this is asynch, thus launch it immediately
        service = OperationService()
        algorithm = adapter_instance.stored_adapter
        if algo_category is None:
            algo_category = dao.get_category_by_id(algorithm.fk_category)
        operation = service.prepare_operations(test_user.id, test_project, algorithm, algo_category,
                                               {}, True, view_model=view_model)[0][0]
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
                                                                   nr_users, current_user.id, users))
                ProjectService().store_project(current_user, True, None, **data)
                ExtremeTestFactory.VALIDATION_DICT[current_user.id] += 1
