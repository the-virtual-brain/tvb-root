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
import json
from time import sleep
import numpy
import pytest
import os.path
import os
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tvb.adapters.analyzers.bct_adapters import BaseBCTModel
from tvb.adapters.analyzers.bct_clustering_adapters import TransitivityBinaryDirected
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex, ValueWrapperIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5, TimeSeriesRegionH5
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex, TimeSeriesRegionIndex
from tvb.basic.profile import TvbProfile
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.load import get_filtered_datatypes, try_get_last_datatype
from tvb.core.entities.model.model_operation import STATUS_FINISHED, Operation, AlgorithmCategory, Algorithm
from tvb.core.entities.model.model_project import User, Project
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import ProjectService
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import Sensors
from tvb.datatypes.surfaces import Surface, CorticalSurface
from tvb.datatypes.time_series import TimeSeries, TimeSeriesRegion
from tvb.simulator.simulator import Simulator
from tvb.tests.framework.adapters.testadapter1 import TestAdapter1
from tvb.tests.framework.core.base_testcase import Base, OperationGroup, DataTypeGroup
from tvb.tests.framework.datatypes.dummy_datatype import DummyDataType
from tvb.tests.framework.datatypes.dummy_datatype_h5 import DummyDataTypeH5
from tvb.tests.framework.datatypes.dummy_datatype_index import DummyDataTypeIndex


def pytest_addoption(parser):
    parser.addoption("--profile", action="store", default="TEST_SQLITE_PROFILE",
                     help="my option: TEST_POSTGRES_PROFILE or TEST_SQLITE_PROFILE")


@pytest.fixture(scope='session', autouse=True)
def profile(request):
    profile = request.config.getoption("--profile")
    TvbProfile.set_profile(profile)
    return profile


@pytest.fixture
def tmph5factory(tmpdir):
    def build(pth='tmp.h5'):
        path = os.path.join(str(tmpdir), pth)
        if os.path.exists(path):
            os.remove(path)
        return path

    return build


@pytest.fixture(scope='session')
def db_engine(tmpdir_factory, profile):
    if profile == TvbProfile.TEST_SQLITE_PROFILE:
        tmpdir = tmpdir_factory.mktemp('tmp')
        path = os.path.join(str(tmpdir), 'tmp.sqlite')
        conn_string = r'sqlite:///' + path
    elif profile == TvbProfile.TEST_POSTGRES_PROFILE:
        conn_string = TvbProfile.current.db.DB_URL
    else:
        raise ValueError('bad test profile {}'.format(profile))

    return create_engine(conn_string)


@pytest.fixture
def session(db_engine):
    Base.metadata.drop_all(db_engine)
    Base.metadata.create_all(db_engine)
    Session = sessionmaker(bind=db_engine)
    s = Session()
    yield s
    s.close()
    Base.metadata.drop_all(db_engine)


@pytest.fixture
def user_factory():
    def build(username='test_user', display_name='test_name', password='test_pass',
              mail='test_mail@tvb.org', validated=True, role='test'):
        """
        Create persisted User entity.
        :returns: User entity after persistence.
        """
        existing_user = dao.get_user_by_name(username)
        if existing_user is not None:
            return existing_user

        user = User(username, display_name, password, mail, validated, role)
        return dao.store_entity(user)

    return build


@pytest.fixture
def project_factory():
    def build(admin, name="TestProject", description='description', users=None):
        """
        Create persisted Project entity, with no linked DataTypes.
        :returns: Project entity after persistence.
        """
        project = dao.get_generic_entity(Project, name, "name")
        if project:
            return project[0]

        if users is None:
            users = []
        data = dict(name=name, description=description, users=users)
        return ProjectService().store_project(admin, True, None, **data)

    return build


@pytest.fixture()
def operation_factory(user_factory, project_factory):
    def build(algorithm=None, test_user=None, test_project=None,
              operation_status=STATUS_FINISHED, parameters="test params", meta=None, range_values=None):
        """
        Create persisted operation.
        :param algorithm: When not None, Simulator.
        :return: Operation entity after persistence.
        """
        if algorithm is None:
            algorithm = dao.get_algorithm_by_module('tvb.adapters.simulator.simulator_adapter', 'SimulatorAdapter')
        if test_user is None:
            test_user = user_factory()
        if test_project is None:
            test_project = project_factory(test_user)

        if meta is None:
            meta = {DataTypeMetaData.KEY_SUBJECT: "John Doe",
                    DataTypeMetaData.KEY_STATE: "RAW_DATA"}
        operation = Operation(test_user.id, test_project.id, algorithm.id, parameters, meta=json.dumps(meta),
                              status=operation_status, range_values=range_values)
        dao.store_entity(operation)
        # Make sure lazy attributes are correctly loaded.
        return dao.get_operation_by_id(operation.id)

    return build


@pytest.fixture()
def connectivity_factory():
    def build(nr_regions=4):
        return Connectivity(
            region_labels=numpy.array(["a"] * nr_regions),
            weights=numpy.zeros((nr_regions, nr_regions)),
            undirected=True,
            tract_lengths=numpy.zeros((nr_regions, nr_regions)),
            centres=numpy.zeros((nr_regions, nr_regions)),
            cortical=numpy.array([True] * nr_regions),
            hemispheres=numpy.array([True] * nr_regions),
            orientations=numpy.zeros((nr_regions, nr_regions)),
            areas=numpy.zeros((nr_regions * nr_regions,)),
            number_of_regions=nr_regions,
            number_of_connections=nr_regions * nr_regions,
            saved_selection=[1, 2, 3]
        )

    return build


@pytest.fixture()
def connectivity_index_factory(connectivity_factory, operation_factory):
    def build(data=4, op=None):
        conn = connectivity_factory(data)
        if op is None:
            op = operation_factory()

        storage_path = FilesHelper().get_project_folder(op.project, str(op.id))
        conn_db = h5.store_complete(conn, storage_path)
        conn_db.fk_from_operation = op.id
        return dao.store_entity(conn_db)

    return build


@pytest.fixture()
def surface_factory():
    def build(nr_vertices=10, valid_for_simulation=True, cortical=False):
        if cortical:
            return CorticalSurface(
                vertices=numpy.zeros((nr_vertices, 3)),
                triangles=numpy.zeros((3, 3), dtype=int),
                vertex_normals=numpy.zeros((nr_vertices, 3)),
                triangle_normals=numpy.zeros((3, 3)),
                number_of_vertices=nr_vertices,
                number_of_triangles=3,
                edge_mean_length=1.0,
                edge_min_length=0.0,
                edge_max_length=2.0,
                zero_based_triangles=False,
                bi_hemispheric=False,
                valid_for_simulations=True
            )

        return Surface(
            vertices=numpy.zeros((nr_vertices, 3)),
            triangles=numpy.zeros((3, 3), dtype=int),
            vertex_normals=numpy.zeros((nr_vertices, 3)),
            triangle_normals=numpy.zeros((3, 3)),
            number_of_vertices=nr_vertices,
            number_of_triangles=3,
            edge_mean_length=1.0,
            edge_min_length=0.0,
            edge_max_length=2.0,
            zero_based_triangles=False,
            bi_hemispheric=False,
            surface_type="surface_cortical",
            valid_for_simulations=valid_for_simulation)

    return build


@pytest.fixture()
def surface_index_factory(surface_factory, operation_factory):
    def build(data=4, op=None):
        surface = surface_factory(data)
        if op is None:
            op = operation_factory()

        storage_path = FilesHelper().get_project_folder(op.project, str(op.id))
        surface_db = h5.store_complete(surface, storage_path)
        surface_db.fk_from_operation = op.id
        return dao.store_entity(surface_db)

    return build


@pytest.fixture()
def region_mapping_factory(surface_factory, connectivity_factory):
    def build(surface=None, connectivity=None):
        if not surface:
            surface = surface_factory(5)
        if not connectivity:
            connectivity = connectivity_factory(2)
        return RegionMapping(
            array_data=numpy.arange(surface.number_of_vertices),
            connectivity=connectivity,
            surface=surface
        )

    return build


@pytest.fixture()
def region_mapping_index_factory(region_mapping_factory, operation_factory):
    def build(op=None):
        region_mapping = region_mapping_factory()
        if op is None:
            op = operation_factory()

        storage_path = FilesHelper().get_project_folder(op.project, str(op.id))
        surface_db = h5.store_complete(region_mapping.surface, storage_path)
        surface_db.fk_from_operation = op.id
        dao.store_entity(surface_db)
        conn_db = h5.store_complete(region_mapping.connectivity, storage_path)
        conn_db.fk_from_operation = op.id
        dao.store_entity(conn_db)
        rm_db = h5.store_complete(region_mapping, storage_path)
        rm_db.fk_from_operation = op.id
        return dao.store_entity(rm_db)

    return build


@pytest.fixture()
def sensors_factory():
    def build(type="EEG", nr_sensors=3):
        return Sensors(
            sensors_type=type,
            labels=numpy.array(["s"] * nr_sensors),
            locations=numpy.ones((nr_sensors, 3)),
            number_of_sensors=nr_sensors,
            has_orientation=True,
            orientations=numpy.zeros((nr_sensors, 3)),
            usable=numpy.array([True] * nr_sensors)
        )

    return build


@pytest.fixture()
def region_simulation_factory(connectivity_factory):
    def build(connectivity=None, simulation_length=100):
        if not connectivity:
            connectivity = connectivity_factory(2)
        return Simulator(connectivity=connectivity,
                         surface=None,
                         simulation_length=simulation_length)

    return build


@pytest.fixture()
def time_series_factory():
    def build(data=None):
        time = numpy.linspace(0, 1000, 4000)

        if data is None:
            data = numpy.zeros((time.size, 1, 3, 1))
            data[:, 0, 0, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 40)
            data[:, 0, 1, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 200)
            data[:, 0, 2, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 100) + numpy.sin(
                2 * numpy.pi * time / 1000.0 * 300)

        return TimeSeries(time=time, data=data, sample_period=1.0 / 4000, sample_period_unit="sec")

    return build


@pytest.fixture()
def time_series_index_factory(time_series_factory, operation_factory):
    def build(data=None, op=None):
        ts = time_series_factory(data)

        if op is None:
            op = operation_factory()

        ts_db = TimeSeriesIndex()
        ts_db.fk_from_operation = op.id
        ts_db.fill_from_has_traits(ts)

        ts_h5_path = h5.path_for_stored_index(ts_db)
        with TimeSeriesH5(ts_h5_path) as f:
            f.store(ts)
            f.sample_rate.store(ts.sample_rate)
            f.nr_dimensions.store(ts.data.ndim)

        ts_db = dao.store_entity(ts_db)
        return ts_db

    return build


@pytest.fixture()
def time_series_region_index_factory(operation_factory):
    def build(connectivity, region_mapping, test_user=None, test_project=None):
        time = numpy.linspace(0, 1000, 4000)
        data = numpy.zeros((time.size, 1, 3, 1))
        data[:, 0, 0, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 40)
        data[:, 0, 1, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 200)
        data[:, 0, 2, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 100) + \
                           numpy.sin(2 * numpy.pi * time / 1000.0 * 300)

        ts = TimeSeriesRegion(time=time, data=data, sample_period=1.0 / 4000, connectivity=connectivity,
                              region_mapping=region_mapping)

        op = operation_factory(test_user=test_user, test_project=test_project)

        ts_db = TimeSeriesRegionIndex()
        ts_db.fk_from_operation = op.id
        ts_db.fill_from_has_traits(ts)

        ts_h5_path = h5.path_for_stored_index(ts_db)
        with TimeSeriesRegionH5(ts_h5_path) as f:
            f.store(ts)
            f.sample_rate.store(ts.sample_rate)
            f.nr_dimensions.store(ts.data.ndim)

        ts_db = dao.store_entity(ts_db)
        return ts_db

    return build


@pytest.fixture()
def dummy_datatype_index_factory(operation_factory):
    def build(row1=None, row2=None, project=None, operation=None, subject=None, state=None):
        data_type = DummyDataType()
        data_type.row1 = row1
        data_type.row2 = row2

        if operation is None:
            operation = operation_factory(test_project=project)

        data_type_index = DummyDataTypeIndex(subject=subject, state=state)
        data_type_index.fk_from_operation = operation.id
        data_type_index.fill_from_has_traits(data_type)

        data_type_h5_path = h5.path_for_stored_index(data_type_index)
        with DummyDataTypeH5(data_type_h5_path) as f:
            f.store(data_type)

        data_type_index = dao.store_entity(data_type_index)
        return data_type_index

    return build


@pytest.fixture()
def value_wrapper_factory():
    def build(test_user, test_project):
        view_model = BaseBCTModel()
        view_model.connectivity = get_filtered_datatypes(test_project.id, ConnectivityIndex, page_size=1)[0][0][2]

        adapter = ABCAdapter.build_adapter_from_class(TransitivityBinaryDirected)
        op = OperationService().fire_operation(adapter, test_user, test_project.id, view_model=view_model)[0]
        # wait for the operation to finish
        tries = 5
        while not op.has_finished and tries > 0:
            sleep(5)
            tries = -1
            op = dao.get_operation_by_id(op.id)

        value_wrapper = try_get_last_datatype(test_project.id, ValueWrapperIndex)
        count = dao.count_datatypes(test_project.id, ValueWrapperIndex)
        assert 1 == count
        return value_wrapper

    return build


@pytest.fixture()
def datatype_measure_factory():
    def build(analyzed_entity, operation, datatype_group, metrics='{"v": 3}'):
        measure = DatatypeMeasureIndex()
        measure.metrics = metrics
        measure.source = analyzed_entity
        measure.fk_from_operation = operation.id
        measure.fk_datatype_group = datatype_group.id
        measure = dao.store_entity(measure)

        return measure

    return build


@pytest.fixture()
def datatype_group_factory(time_series_index_factory, datatype_measure_factory, project_factory, user_factory,
                           operation_factory):
    def build(subject="Datatype Factory User", state="RAW_DATA", project=None):

        # there store the name and the (hi, lo, step) value of the range parameters
        range_1 = ["row1", [1, 2, 10]]
        range_2 = ["row2", [0.1, 0.3, 0.5]]

        # there are the actual numbers in the interval
        range_values_1 = [1, 3, 5, 7, 9]
        range_values_2 = [0.1, 0.4]

        user = user_factory()

        if project is None:
            project = project_factory(user)

        # Create an algorithm
        alg_category = AlgorithmCategory('one', True)
        dao.store_entity(alg_category)
        ad = Algorithm(IntrospectionRegistry.SIMULATOR_MODULE, IntrospectionRegistry.SIMULATOR_CLASS,
                       alg_category.id)
        algorithm = dao.get_algorithm_by_module(IntrospectionRegistry.SIMULATOR_MODULE,
                                                IntrospectionRegistry.SIMULATOR_CLASS)

        if algorithm is None:
            algorithm = dao.store_entity(ad)

        # Create meta
        meta = {DataTypeMetaData.KEY_SUBJECT: "Datatype Factory User",
                DataTypeMetaData.KEY_STATE: "RAW_DATA"}

        # Create operation
        operation = operation_factory(algorithm=algorithm, test_user=user, test_project=project, meta=meta)

        group = OperationGroup(project.id, ranges=[json.dumps(range_1), json.dumps(range_2)])
        group = dao.store_entity(group)
        group_ms = OperationGroup(project.id, ranges=[json.dumps(range_1), json.dumps(range_2)])
        group_ms = dao.store_entity(group_ms)

        datatype_group = DataTypeGroup(group, subject=subject, state=state, operation_id=operation.id)
        datatype_group.no_of_ranges = 2
        datatype_group.count_results = 10

        datatype_group = dao.store_entity(datatype_group)

        dt_group_ms = DataTypeGroup(group_ms, subject=subject, state=state, operation_id=operation.id)
        dao.store_entity(dt_group_ms)

        # Now create some data types and add them to group
        for range_val1 in range_values_1:
            for range_val2 in range_values_2:
                op = Operation(user.id, project.id, algorithm.id, 'test parameters',
                               meta=json.dumps(meta), status=STATUS_FINISHED,
                               range_values=json.dumps({range_1[0]: range_val1,
                                                        range_2[0]: range_val2}))
                op.fk_operation_group = group.id
                op = dao.store_entity(op)
                datatype = time_series_index_factory(op=op)
                datatype.number1 = range_val1
                datatype.number2 = range_val2
                datatype.fk_datatype_group = datatype_group.id
                datatype.operation_id = op.id
                dao.store_entity(datatype)

                op_ms = Operation(user.id, project.id, algorithm.id, 'test parameters',
                                  meta=json.dumps(meta), status=STATUS_FINISHED,
                                  range_values=json.dumps({range_1[0]: range_val1,
                                                           range_2[0]: range_val2}))
                op_ms.fk_operation_group = group_ms.id
                op_ms = dao.store_entity(op_ms)
                datatype_measure_factory(datatype, op_ms, dt_group_ms)

        return datatype_group

    return build


@pytest.fixture()
def test_adapter_factory():
    def build(adapter_class=TestAdapter1):

        all_categories = dao.get_algorithm_categories()
        algo_category_id = all_categories[0].id

        stored_adapter = Algorithm(adapter_class.__module__, adapter_class.__name__, algo_category_id,
                                   adapter_class.get_group_name(), adapter_class.get_group_description(),
                                   adapter_class.get_ui_name(), adapter_class.get_ui_description(),
                                   adapter_class.get_ui_subsection(), datetime.datetime.now())
        adapter_inst = adapter_class()

        adapter_form = adapter_inst.get_form()
        required_datatype = adapter_form.get_required_datatype()
        if required_datatype is not None:
            required_datatype = required_datatype.__name__
        filters = adapter_form.get_filters()
        if filters is not None:
            filters = filters.to_json()

        stored_adapter.required_datatype = required_datatype
        stored_adapter.datatype_filter_filter = filters
        stored_adapter.parameter_name = adapter_form.get_input_name()
        stored_adapter.outputlist = str(adapter_inst.get_output())

        inst_from_db = dao.get_algorithm_by_module(adapter_class.__module__, adapter_class.__name__)
        if inst_from_db is not None:
            stored_adapter.id = inst_from_db.id

        return dao.store_entity(stored_adapter, inst_from_db is not None)

    return build


@pytest.fixture()
def local_connectivity_index_factory(surface_factory, operation_factory):
    def build(op=None):
        surface = surface_factory(cortical=True)
        lconn = LocalConnectivity()
        lconn.surface = surface
        if op is None:
            op = operation_factory()

        storage_path = FilesHelper().get_project_folder(op.project, str(op.id))
        surface_db = h5.store_complete(surface, storage_path)
        surface_db.fk_from_operation = op.id
        dao.store_entity(surface_db)

        lconn_db = h5.store_complete(lconn, storage_path)
        lconn_db.fk_from_operation = op.id
        return dao.store_entity(lconn_db), lconn

    return build
