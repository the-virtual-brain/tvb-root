# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
import json
import numpy
import pytest
import os.path
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tvb.datatypes.graph import Covariance
from tvb.datatypes.time_series import TimeSeries

from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5
from tvb.adapters.datatypes.db.graph import CovarianceIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.entities.model.model_operation import STATUS_FINISHED, Operation
from tvb.core.entities.model.model_project import User, Project
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.neocom.h5 import h5_file_for_index
from tvb.core.services.project_service import ProjectService
from tvb.core.neotraits.db import Base
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import Sensors
from tvb.datatypes.surfaces import Surface, CorticalSurface
from tvb.simulator.simulator import Simulator
from tvb.tests.framework.core.base_testcase import TvbProfile

def pytest_addoption(parser):
    parser.addoption("--profile", action="store", default="TEST_SQLITE_PROFILE",
                     help="my option: TEST_POSTGRES_PROFILE or TEST_SQLITE_PROFILE")


@pytest.fixture(scope='session')
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
    if profile == 'TEST_SQLITE_PROFILE':
        tmpdir = tmpdir_factory.mktemp('tmp')
        path = os.path.join(str(tmpdir), 'tmp.sqlite')
        conn_string = r'sqlite:///' + path
    elif profile == 'TEST_POSTGRES_PROFILE':
        conn_string = 'postgresql+psycopg2://tvb:tvb23@localhost:5432/tvb'
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
    def build(username='test_user', password='test_pass',
              mail='test_mail@tvb.org', validated=True, role='test'):
        """
        Create persisted User entity.
        :returns: User entity after persistence.
        """
        existing_user = dao.get_user_by_name(username)
        if existing_user is not None:
            return existing_user

        user = User(username, password, mail, validated, role)
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
              operation_status=STATUS_FINISHED, parameters="test params"):
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

        meta = {DataTypeMetaData.KEY_SUBJECT: "John Doe",
                DataTypeMetaData.KEY_STATE: "RAW_DATA"}
        operation = Operation(test_user.id, test_project.id, algorithm.id, parameters, meta=json.dumps(meta),
                              status=operation_status)
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
                split_triangles=numpy.arange(0),
                number_of_split_slices=1,
                split_slices=dict(),
                bi_hemispheric=False,
                # surface_type="surface",
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
            split_triangles=numpy.arange(0),
            number_of_split_slices=1,
            split_slices=dict(),
            bi_hemispheric=False,
            surface_type="surface_cortical",
            valid_for_simulations=valid_for_simulation)

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
def time_series_factory(operation_factory, session):
    def build():
        time = numpy.linspace(0, 1000, 4000)
        data = numpy.zeros((time.size, 1, 3, 1))
        data[:, 0, 0, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 40)
        data[:, 0, 1, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 200)
        data[:, 0, 2, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 100) + \
                           numpy.sin(2 * numpy.pi * time / 1000.0 * 300)

        ts = TimeSeries(time=time, data=data, sample_period=1.0 / 4000)

        op = operation_factory()

        ts_db = TimeSeriesIndex()
        ts_db.fk_from_operation = op.id
        ts_db.fill_from_has_traits(ts)

        ts_h5_path = h5.path_for_stored_index(ts_db)
        with TimeSeriesH5(ts_h5_path) as f:
            f.store(ts)

        session.add(ts_db)
        session.commit()
        return ts_db
    return build

@pytest.fixture()
def covariance_factory(time_series_factory, operation_factory, session):
    def build():
        ts_index = time_series_factory()

        ts_h5 = h5_file_for_index(ts_index)
        ts = TimeSeries()
        ts_h5.load_into(ts)
        ts_h5.close()

        data_shape = ts.data.shape

        result_shape = (data_shape[2], data_shape[2], data_shape[1], data_shape[3])
        result = numpy.zeros(result_shape)

        for mode in range(data_shape[3]):
            for var in range(data_shape[1]):
                data = ts_h5.data[:, var, :, mode]
                data = data - data.mean(axis=0)[numpy.newaxis, 0]
                result[:, :, var, mode] = numpy.cov(data.T)

        covariance = Covariance(source=ts, array_data=result)

        op = operation_factory()

        covariance_db = CovarianceIndex()
        covariance_db.fk_from_operation = op.id
        covariance_db.fill_from_has_traits(covariance)

        covariance_h5_path = h5.path_for_stored_index(covariance_db)
        with TimeSeriesH5(covariance_h5_path) as f:
            f.store(ts)

        session.add(covariance_db)
        session.commit()
        return covariance_db

    return build
