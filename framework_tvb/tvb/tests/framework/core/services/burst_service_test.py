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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import os
from uuid import UUID
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.services.burst_service import BurstService
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.entities.model.model_burst import *
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.algorithm_service import AlgorithmService, GenericAttributes
from tvb.core.services.project_service import ProjectService
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.datatypes.datatype2 import Datatype2


class TestBurstService(BaseTestCase):
    """
    Test the service layer for BURST PAGE. We can't have this transactional since
    we launch operations in different threads and the transactional operator only rolls back 
    sessions bounded to the current thread transaction.
    """
    burst_service = BurstService()
    sim_algorithm = AlgorithmService().get_algorithm_by_module_and_class(IntrospectionRegistry.SIMULATOR_MODULE,
                                                                         IntrospectionRegistry.SIMULATOR_CLASS)

    def setup_method(self):
        """
        Sets up the environment for running the tests;
        cleans the database before testing and saves config file;
        creates a test user, a test project;
        creates burst, flow, operation and workflow services

        """
        self.clean_database()
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user)

    def teardown_method(self):
        """
        Remove project folders and clean up database.
        """
        FilesHelper().remove_project_structure(self.test_project.name)
        self.clean_database()

    def test_clone_burst_configuration(self):
        """
        Test that all the major attributes are the same after a clone burst but the
        id of the cloned one is None.
        """
        first_burst = TestFactory.store_burst(self.test_project.id)
        cloned_burst = first_burst.clone()
        self._compare_bursts(first_burst, cloned_burst)
        assert cloned_burst.name == first_burst.name, 'Cloned burst should have the same name'
        assert cloned_burst.id is None, 'id should be none for cloned entry.'

    def test_store_burst_config(self):
        """
        Test that a burst entity is properly stored in db.
        """
        burst_config = TestFactory.store_burst(self.test_project.id)
        assert burst_config.id is not None, 'Burst was not stored properly.'
        stored_entity = dao.get_burst_by_id(burst_config.id)
        assert stored_entity is not None, 'Burst was not stored properly.'
        self._compare_bursts(burst_config, stored_entity)

    def _compare_bursts(self, first_burst, second_burst):
        """
        Compare that all important attributes are the same between two bursts. (name, project id and status)
        """
        assert first_burst.name == second_burst.name, "Names not equal for bursts."
        assert first_burst.fk_project == second_burst.fk_project, "Projects not equal for bursts."
        assert first_burst.status == second_burst.status, "Statuses not equal for bursts."
        assert first_burst.range1 == second_burst.range1, "Statuses not equal for bursts."
        assert first_burst.range2 == second_burst.range2, "Statuses not equal for bursts."

    def test_getavailablebursts_none(self):
        """
        Test that an empty list is returned if no data is available in db.
        """
        bursts = self.burst_service.get_available_bursts(self.test_project.id)
        assert bursts == [], "Unexpected result returned : %s" % (bursts,)

    def test_get_available_bursts_happy(self):
        """
        Test that all the correct burst are returned for the given project.
        """
        project = Project("second_test_proj", self.test_user.id, "description")
        second_project = dao.store_entity(project)
        test_project_bursts = [TestFactory.store_burst(self.test_project.id).id for _ in range(4)]
        second_project_bursts = [TestFactory.store_burst(second_project.id).id for _ in range(3)]
        returned_test_project_bursts = [burst.id for burst in
                                        self.burst_service.get_available_bursts(self.test_project.id)]
        returned_second_project_bursts = [burst.id for burst in
                                          self.burst_service.get_available_bursts(second_project.id)]
        assert len(test_project_bursts) == len(returned_test_project_bursts), \
            "Incorrect bursts retrieved for project %s." % self.test_project
        assert len(second_project_bursts) == len(returned_second_project_bursts), \
            "Incorrect bursts retrieved for project %s." % second_project
        assert set(second_project_bursts) == set(returned_second_project_bursts), \
            "Incorrect bursts retrieved for project %s." % second_project
        assert set(test_project_bursts) == set(returned_test_project_bursts), \
            "Incorrect bursts retrieved for project %s." % self.test_project

    def test_rename_burst(self, operation_factory):
        """
        Test that renaming of a burst functions properly.
        """
        operation = operation_factory()
        burst_config = TestFactory.store_burst(self.test_project.id, operation)
        self.burst_service.rename_burst(burst_config.id, "new_burst_name")
        loaded_burst = dao.get_burst_by_id(burst_config.id)
        assert loaded_burst.name == "new_burst_name", "Burst was not renamed properly."

    def test_burst_delete_with_project(self):
        """
        Test that on removal of a project all burst related data is cleared.
        """
        TestFactory.store_burst(self.test_project.id)
        ProjectService().remove_project(self.test_project.id)
        self._check_burst_removed()

    def test_load_burst_configuration(self):
        """
        Test that loads the burst configuration based non the stored config id
        """
        stored_burst = TestFactory.store_burst(self.test_project.id)
        burst_config = self.burst_service.load_burst_configuration(stored_burst.id)
        assert burst_config.id == stored_burst.id, "The loaded burst does not have the same ID"

    def test_update_simulation_fields(self, tmph5factory):
        """
        Test that updates the simulation fields of the burst
        """
        stored_burst = TestFactory.store_burst(self.test_project.id)

        connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project)
        op = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)
        simulation = SimulatorAdapterModel()
        simulation.connectivity = UUID(connectivity.gid)

        burst_config = self.burst_service.update_simulation_fields(stored_burst.id, op.id, simulation.gid)
        assert burst_config.id == stored_burst.id, "The loaded burst does not have the same ID"
        assert burst_config.fk_simulation == op.id, "The loaded burst does not have the fk simulation that it was given"
        assert burst_config.simulator_gid == simulation.gid.hex, "The loaded burst does not have the simulation gid that it was given"

    def test_prepare_name(self):
        """
        Test prepare burst name
        """
        stored_burst = TestFactory.store_burst(self.test_project.id)
        simulation_tuple = self.burst_service.prepare_name(stored_burst, self.test_project.id)
        assert simulation_tuple[0] == 'simulation_' + str(dao.get_number_of_bursts(self.test_project.id) + 1), \
            "The default simulation name is not the defined one"

        busrt_test_name = "Burst Test Name"
        stored_burst.name = busrt_test_name
        stored_burst = dao.store_entity(stored_burst)
        simulation_tuple = self.burst_service.prepare_name(stored_burst, self.test_project.id)
        assert simulation_tuple[0] == busrt_test_name, "The burst name is not the given one"

    def test_prepare_burst_for_pse(self):
        """
        Test prepare burst for pse
        """
        burst = BurstConfiguration(self.test_project.id)
        assert burst.fk_metric_operation_group == None, "The fk for the metric operation group is not None"
        assert burst.fk_operation_group == None, "The fk for the operation group is not None"
        assert burst.operation_group == None, "The operation group is not None"

        pse_burst = self.burst_service.prepare_burst_for_pse(burst)
        assert pse_burst.metric_operation_group != None, "The fk for the operation group is None"
        assert pse_burst.operation_group != None, "The operation group is None"

    def _check_burst_removed(self):
        """
        Test that a burst was properly removed. This means checking that the burst entity,
        any workflow steps and any datatypes resulted from the burst are also removed.
        """
        remaining_bursts = dao.get_bursts_for_project(self.test_project.id)
        assert 0 == len(remaining_bursts), "Burst was not deleted"
        ops_number = dao.get_operation_numbers(self.test_project.id)[0]
        assert 0 == ops_number, "Operations were not deleted."
        datatypes = dao.get_datatypes_in_project(self.test_project.id)
        assert 0 == len(datatypes)

        datatype1_stored = self.count_all_entities(Datatype1)
        datatype2_stored = self.count_all_entities(Datatype2)
        assert 0 == datatype1_stored, "Specific datatype entries for DataType1 were not deleted."
        assert 0 == datatype2_stored, "Specific datatype entries for DataType2 were not deleted."

    def test_prepare_indexes_for_simulation_results(self, time_series_factory, operation_factory, simulator_factory):
        ts_1 = time_series_factory()
        ts_2 = time_series_factory()
        ts_3 = time_series_factory()

        operation = operation_factory(test_user=self.test_user, test_project=self.test_project)
        sim_folder, sim_gid = simulator_factory(op=operation)

        path_1 = os.path.join(sim_folder, "Time_Series_{}.h5".format(ts_1.gid.hex))
        path_2 = os.path.join(sim_folder, "Time_Series_{}.h5".format(ts_2.gid.hex))
        path_3 = os.path.join(sim_folder, "Time_Series_{}.h5".format(ts_3.gid.hex))

        with TimeSeriesH5(path_1) as f:
            f.store(ts_1)
            f.store_generic_attributes(GenericAttributes())

        with TimeSeriesH5(path_2) as f:
            f.store(ts_2)
            f.store_generic_attributes(GenericAttributes())

        with TimeSeriesH5(path_3) as f:
            f.store(ts_3)
            f.store_generic_attributes(GenericAttributes())

        burst_configuration = BurstConfiguration(self.test_project.id)
        burst_configuration.fk_simulation = operation.id
        burst_configuration.simulator_gid = operation.view_model_gid
        burst_configuration = dao.store_entity(burst_configuration)

        file_names = [path_1, path_2, path_3]
        ts_datatypes = [ts_1, ts_2, ts_3]
        indexes = self.burst_service.prepare_indexes_for_simulation_results(operation, file_names, burst_configuration)

        for i in range(len(indexes)):
            assert indexes[i].gid == ts_datatypes[i].gid.hex, "Gid was not set correctly on index."
            assert indexes[i].sample_period == ts_datatypes[i].sample_period
            assert indexes[i].sample_period_unit == ts_datatypes[i].sample_period_unit
            assert indexes[i].sample_rate == ts_datatypes[i].sample_rate

