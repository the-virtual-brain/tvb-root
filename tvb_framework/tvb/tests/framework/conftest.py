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

import copy
import json
import os
import os.path
import uuid
import numpy
import pytest
from datetime import datetime
from time import sleep
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tvb.adapters.analyzers.bct_adapters import BaseBCTModel
from tvb.adapters.analyzers.bct_clustering_adapters import TransitivityBinaryDirected
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex, ValueWrapperIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex, TimeSeriesRegionIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5, TimeSeriesRegionH5
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapterModel
from tvb.basic.profile import TvbProfile
from tvb.basic.neotraits.api import Range
from tvb.config import SIMULATOR_MODULE, SIMULATOR_CLASS, TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS
from tvb.config import MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS
from tvb.core.entities.file.simulator.datatype_measure_h5 import DatatypeMeasure, DatatypeMeasureH5
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.core.services.burst_service import BurstService
from tvb.core.services.simulator_service import SimulatorService
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.simulator.view_model import TemporalAverageViewModel, CortexViewModel
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.load import get_filtered_datatypes, try_get_last_datatype
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_operation import STATUS_FINISHED, Operation, Algorithm
from tvb.core.entities.model.model_project import User, Project
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import ProjectService
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import Sensors, SensorsEEG
from tvb.datatypes.surfaces import Surface, CorticalSurface, SurfaceTypesEnum
from tvb.datatypes.time_series import TimeSeries, TimeSeriesRegion
from tvb.simulator.simulator import Simulator
from tvb.storage.storage_interface import StorageInterface
from tvb.tests.framework.adapters.dummy_adapter1 import DummyAdapter1
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
        data = dict(name=name, description=description, users=users, max_operation_size=None, disable_imports=False)
        return ProjectService().store_project(admin, True, None, **data)

    return build


@pytest.fixture()
def operation_factory(user_factory, project_factory, connectivity_factory):
    def build(test_user=None, test_project=None, is_simulation=False, store_vm=False,
              operation_status=STATUS_FINISHED, range_values=None, conn_gid=None):
        """
        Create persisted operation with a ViewModel stored
        :return: Operation entity after persistence.
        """
        if test_user is None:
            test_user = user_factory()
        if test_project is None:
            test_project = project_factory(test_user)

        vm_gid = uuid.uuid4()
        view_model = None

        if is_simulation:
            algorithm = dao.get_algorithm_by_module(SIMULATOR_MODULE, SIMULATOR_CLASS)
            if store_vm:
                adapter = ABCAdapter.build_adapter(algorithm)
                view_model = adapter.get_view_model_class()()
                view_model.connectivity = connectivity_factory(4).gid if conn_gid is None else conn_gid
                vm_gid = view_model.gid
        else:
            algorithm = dao.get_algorithm_by_module(TVB_IMPORTER_MODULE, TVB_IMPORTER_CLASS)
            if store_vm:
                adapter = ABCAdapter.build_adapter(algorithm)
                view_model = adapter.get_view_model_class()()
                view_model.data_file = "."
                vm_gid = view_model.gid

        operation = Operation(vm_gid.hex, test_user.id, test_project.id, algorithm.id,
                              status=operation_status, range_values=range_values)
        dao.store_entity(operation)

        if store_vm:
            op_folder = StorageInterface().get_project_folder(test_project.name, str(operation.id))
            h5.store_view_model(view_model, op_folder)

        # Make sure lazy attributes are correctly loaded.
        return dao.get_operation_by_id(operation.id)

    return build


@pytest.fixture()
def operation_from_existing_op_factory(operation_factory):
    def build(existing_op_id):
        op = dao.get_operation_by_id(existing_op_id)
        project = dao.get_project_by_id(op.fk_launched_in)
        user = dao.get_user_by_id(op.fk_launched_by)

        return operation_factory(test_user=user, test_project=project), project.id

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
    def build(data=4, op=None, conn=None):
        if conn is None:
            conn = connectivity_factory(data)
        if op is None:
            op = operation_factory()

        conn_db = h5.store_complete(conn, op.id, op.project.name)
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
            surface_type=SurfaceTypesEnum.CORTICAL_SURFACE.value,
            valid_for_simulations=valid_for_simulation)

    return build


@pytest.fixture()
def surface_index_factory(surface_factory, operation_factory):
    def build(data=4, op=None, cortical=False, surface=None):
        if not surface:
            surface = surface_factory(data, cortical=cortical)
        if op is None:
            op = operation_factory()

        surface_db = h5.store_complete(surface, op.id, op.project.name)
        surface_db.fk_from_operation = op.id
        return dao.store_entity(surface_db), surface

    return build


@pytest.fixture()
def region_mapping_factory(surface_factory, connectivity_factory):
    def build(surface=None, connectivity=None):
        if not surface:
            surface = surface_factory(5, cortical=True)
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
    def build(op=None, conn_gid=None, surface_gid=None, region_mapping=None):
        if not region_mapping:
            region_mapping = region_mapping_factory()
        if op is None:
            op = operation_factory()

        if not surface_gid:
            surface_db = h5.store_complete(region_mapping.surface, op.id, op.project.name)
            surface_db.fk_from_operation = op.id
            dao.store_entity(surface_db)
        else:
            region_mapping.surface.gid = uuid.UUID(surface_gid)
        if not conn_gid:
            conn_db = h5.store_complete(region_mapping.connectivity, op.id, op.project.name)
            conn_db.fk_from_operation = op.id
            dao.store_entity(conn_db)
        else:
            region_mapping.connectivity.gid = uuid.UUID(conn_gid)
        rm_db = h5.store_complete(region_mapping, op.id, op.project.name)
        rm_db.fk_from_operation = op.id
        return dao.store_entity(rm_db)

    return build


@pytest.fixture()
def connectivity_measure_index_factory():
    def build(conn, op, project):
        conn_measure = ConnectivityMeasure()
        conn_measure.connectivity = h5.load_from_index(conn)
        conn_measure.array_data = numpy.ones(conn.number_of_regions)

        conn_measure_db = h5.store_complete(conn_measure, op.id, project.name)
        conn_measure_db.fk_from_operation = op.id
        return dao.store_entity(conn_measure_db)

    return build


@pytest.fixture()
def sensors_factory():
    def build(type="EEG", nr_sensors=3):
        if type == "EEG":
            return SensorsEEG(
                labels=numpy.array(["s"] * nr_sensors),
                locations=numpy.ones((nr_sensors, 3)),
                number_of_sensors=nr_sensors,
                has_orientation=True,
                orientations=numpy.zeros((nr_sensors, 3)),
                usable=numpy.array([True] * nr_sensors)
            )
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
def sensors_index_factory(sensors_factory, operation_factory):
    def build(type="EEG", nr_sensors=3, op=None):
        sensors = sensors_factory(type, nr_sensors)
        if op is None:
            op = operation_factory()

        sensors_db = h5.store_complete(sensors, op.id, op.project.name)
        sensors_db.fk_from_operation = op.id
        return dao.store_entity(sensors_db), sensors

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
    def build(ts=None, data=None, op=None):
        if ts is None:
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
            f.store_generic_attributes(GenericAttributes())
            f.store_references(ts)

        ts_db = dao.store_entity(ts_db)
        return ts_db

    return build


@pytest.fixture()
def time_series_region_factory():
    def build(connectivity, region_mapping):
        time = numpy.linspace(0, 1000, 4000)
        data = numpy.zeros((time.size, 1, 3, 1))
        data[:, 0, 0, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 40)
        data[:, 0, 1, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 200)
        data[:, 0, 2, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 100) + \
                           numpy.sin(2 * numpy.pi * time / 1000.0 * 300)

        ts = TimeSeriesRegion(time=time, data=data, sample_period=1.0 / 4000, connectivity=connectivity,
                              region_mapping=region_mapping)
        return ts

    return build


@pytest.fixture()
def time_series_region_index_factory(operation_factory, time_series_region_factory):
    def build(connectivity, region_mapping, ts=None, test_user=None, test_project=None, op=None):
        if ts is None:
            ts = time_series_region_factory(connectivity, region_mapping)

        if not op:
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
        op = OperationService().fire_operation(adapter, test_user, test_project.id, view_model=view_model)
        # wait for the operation to finish
        tries = 5
        while not op.has_finished and tries > 0:
            sleep(5)
            tries = tries - 1
            op = dao.get_operation_by_id(op.id)

        value_wrapper = try_get_last_datatype(test_project.id, ValueWrapperIndex)
        count = dao.count_datatypes(test_project.id, ValueWrapperIndex)
        assert 1 == count
        return value_wrapper

    return build


@pytest.fixture()
def datatype_measure_factory():
    def build(analyzed_entity_index, analyzed_entity, operation, datatype_group, metrics='{"v": 3}'):
        measure = DatatypeMeasureIndex()
        measure.metrics = metrics
        measure.source = analyzed_entity_index
        measure.fk_from_operation = operation.id
        measure.fk_datatype_group = datatype_group.id
        measure = dao.store_entity(measure)

        dm = DatatypeMeasure(analyzed_datatype=analyzed_entity, metrics=json.loads(metrics))
        dm_path = h5.path_for_stored_index(measure)

        with DatatypeMeasureH5(dm_path) as dm_h5:
            dm_h5.store(dm)
            dm_h5.store_generic_attributes(GenericAttributes())

        return measure

    return build


@pytest.fixture()
def datatype_group_factory(connectivity_factory, time_series_index_factory, time_series_factory,
                           time_series_region_factory, datatype_measure_factory, project_factory, user_factory,
                           operation_factory, time_series_region_index_factory, region_mapping_factory, surface_factory,
                           connectivity_index_factory, region_mapping_index_factory, surface_index_factory):
    def build(project=None, store_vm=False, use_time_series_region=False, status=STATUS_FINISHED):
        # there store the name and the (hi, lo, step) value of the range parameters
        range_1 = ["row1", [1, 2, 6]]
        range_2 = ["row2", [0.1, 0.3, 0.5]]
        # there are the actual numbers in the interval
        range_values_1 = [1, 3, 5]
        range_values_2 = [0.1, 0.4]

        user = user_factory()
        if project is None:
            project = project_factory(user)

        connectivity = connectivity_factory(4)
        if use_time_series_region:
            operation = operation_factory(test_project=project)
            connectivity_index_factory(op=operation, conn=connectivity)

            operation2 = operation_factory(test_project=project)
            surface = surface_factory()
            surface_index_factory(op=operation2, surface=surface)

            operation3 = operation_factory(test_project=project)
            region_mapping = region_mapping_factory(surface=surface, connectivity=connectivity)
            region_mapping_index_factory(op=operation3, conn_gid=connectivity.gid.hex, surface_gid=surface.gid.hex,
                                         region_mapping=region_mapping)

        algorithm = dao.get_algorithm_by_module(SIMULATOR_MODULE, SIMULATOR_CLASS)
        adapter = ABCAdapter.build_adapter(algorithm)
        if store_vm:
            view_model = adapter.get_view_model_class()()
            view_model.connectivity = connectivity.gid
        else:
            view_model = None

        algorithm_ms = dao.get_algorithm_by_module(MEASURE_METRICS_MODULE, MEASURE_METRICS_CLASS)
        adapter = ABCAdapter.build_adapter(algorithm_ms)
        view_model_ms = adapter.get_view_model_class()()

        op_group = OperationGroup(project.id, ranges=[json.dumps(range_1), json.dumps(range_2)])
        op_group = dao.store_entity(op_group)
        op_group_ms = OperationGroup(project.id, ranges=[json.dumps(range_1), json.dumps(range_2)])
        op_group_ms = dao.store_entity(op_group_ms)

        datatype_group = DataTypeGroup(op_group, state="RAW_DATA")
        datatype_group.no_of_ranges = 2
        datatype_group.count_results = 6
        datatype_group = dao.store_entity(datatype_group)

        dt_group_ms = DataTypeGroup(op_group_ms, state="RAW_DATA")
        dt_group_ms.no_of_ranges = 2
        dt_group_ms.count_results = 6
        dao.store_entity(dt_group_ms)

        # Now create some data types and add them to group
        for range_val1 in range_values_1:
            for range_val2 in range_values_2:

                view_model_gid = uuid.uuid4()
                view_model_ms_gid = uuid.uuid4()

                op = Operation(view_model_gid.hex, user.id, project.id, algorithm.id,
                               status=status, op_group_id=op_group.id,
                               range_values=json.dumps({range_1[0]: range_val1,
                                                        range_2[0]: range_val2}))
                op = dao.store_entity(op)
                if use_time_series_region:
                    ts = time_series_region_factory(connectivity=connectivity, region_mapping=region_mapping)
                    ts_index = time_series_region_index_factory(ts=ts, connectivity=connectivity,
                                                                region_mapping=region_mapping, test_user=user,
                                                                test_project=project, op=op)
                else:
                    ts = time_series_factory()
                    ts_index = time_series_index_factory(ts=ts, op=op)
                ts_index.fk_datatype_group = datatype_group.id
                dao.store_entity(ts_index)

                op_ms = Operation(view_model_ms_gid.hex, user.id, project.id, algorithm.id,
                                  status=STATUS_FINISHED, op_group_id=op_group_ms.id,
                                  range_values=json.dumps({range_1[0]: range_val1,
                                                           range_2[0]: range_val2}))
                op_ms = dao.store_entity(op_ms)
                datatype_measure_factory(ts_index, ts, op_ms, dt_group_ms)

                if store_vm:
                    view_model = copy.deepcopy(view_model)
                    view_model.gid = view_model_gid
                    op_path = StorageInterface().get_project_folder(project.name, str(op.id))
                    h5.store_view_model(view_model, op_path)

                    view_model_ms.gid = view_model_ms_gid
                    view_model_ms.time_series = ts_index.gid
                    view_model_ms = copy.deepcopy(view_model_ms)    # deepcopy only after a TS is set
                    op_ms_path = StorageInterface().get_project_folder(project.name, str(op_ms.id))
                    h5.store_view_model(view_model_ms, op_ms_path)

                if not datatype_group.fk_from_operation:
                    # Mark first operation ID
                    datatype_group.fk_from_operation = op.id
                    dt_group_ms.fk_from_operation = op_ms.id
                    datatype_group = dao.store_entity(datatype_group)
                    dt_group_ms = dao.store_entity(dt_group_ms)

        return datatype_group, dt_group_ms

    return build


@pytest.fixture()
def test_adapter_factory():
    def build(adapter_class=DummyAdapter1):

        all_categories = dao.get_algorithm_categories()
        algo_category_id = all_categories[0].id

        stored_adapter = Algorithm(adapter_class.__module__, adapter_class.__name__, algo_category_id,
                                   adapter_class.get_group_name(), adapter_class.get_group_description(),
                                   adapter_class.get_ui_name(), adapter_class.get_ui_description(),
                                   adapter_class.get_ui_subsection(), datetime.now())
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

        surface_db = h5.store_complete(surface, op.id, op.project.name)
        surface_db.fk_from_operation = op.id
        dao.store_entity(surface_db)

        lconn_db = h5.store_complete(lconn, op.id, op.project.name)
        lconn_db.fk_from_operation = op.id
        return dao.store_entity(lconn_db), lconn

    return build


@pytest.fixture()
def simulator_factory(connectivity_index_factory, operation_factory, region_mapping_index_factory):
    def build(user=None, project=None, op=None, nr_regions=76, monitor=TemporalAverageViewModel(), with_surface=False,
              conn_gid=None):
        model = SimulatorAdapterModel()
        model.monitors = [monitor]
        if not op:
            op = operation_factory(test_user=user, test_project=project)
        if conn_gid:
            model.connectivity = conn_gid
        if not with_surface and not conn_gid:
            model.connectivity = connectivity_index_factory(nr_regions, op).gid
        model.simulation_length = 100
        if with_surface:
            rm_idx = region_mapping_index_factory()
            model.connectivity = rm_idx.fk_connectivity_gid
            model.surface = CortexViewModel()
            model.surface.surface_gid = rm_idx.fk_surface_gid
            model.surface.region_mapping_data = rm_idx.gid
            model.simulation_length = 10
        storage_path = StorageInterface().get_project_folder(op.project.name, str(op.id))
        h5.store_view_model(model, storage_path)

        return storage_path, model.gid

    return build


@pytest.fixture()
def pse_burst_configuration_factory():
    def build(project):
        range_1 = ["row1", [1, 2, 10]]
        range_2 = ["row2", [0.1, 0.3, 0.5]]

        group = OperationGroup(project.id, ranges=[json.dumps(range_1), json.dumps(range_2)])
        group = dao.store_entity(group)
        group_ms = OperationGroup(project.id, ranges=[json.dumps(range_1), json.dumps(range_2)])
        group_ms = dao.store_entity(group_ms)

        datatype_group = DataTypeGroup(group)
        datatype_group.no_of_ranges = 2
        datatype_group.count_results = 10
        dao.store_entity(datatype_group)

        dt_group_ms = DataTypeGroup(group_ms)
        dao.store_entity(dt_group_ms)

        burst = BurstConfiguration(project.id, name='test_burst')
        burst.simulator_gid = uuid.uuid4().hex
        burst.fk_operation_group = group.id
        burst.fk_metric_operation_group = group_ms.id
        burst = dao.store_entity(burst)
        return burst

    return build


@pytest.fixture()
def simulation_launch(connectivity_index_factory):
    def build(test_user, test_project, simulation_length=10, is_group=False):
        model = SimulatorAdapterModel()
        model.connectivity = connectivity_index_factory().gid
        model.simulation_length = simulation_length
        burst = BurstConfiguration(test_project.id, name="Sim " + str(datetime.now()))
        burst.start_time = datetime.now()
        algorithm = dao.get_algorithm_by_module(SIMULATOR_MODULE, SIMULATOR_CLASS)
        service = SimulatorService()
        if is_group:
            range_param = RangeParameter("conduction_speed", float, Range(lo=50.0, hi=100.0, step=20.0))
            burst.range1 = range_param.to_json()
            burst = BurstService().prepare_burst_for_pse(burst)
            op = service.async_launch_and_prepare_pse(burst, test_user, test_project, algorithm,
                                                      range_param, None, model)
        else:
            dao.store_entity(burst)
            op = service.async_launch_and_prepare_simulation(burst, test_user, test_project, algorithm, model)
        return op

    return build
