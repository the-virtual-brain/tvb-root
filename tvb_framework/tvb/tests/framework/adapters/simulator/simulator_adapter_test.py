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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import tvb_data.surfaceData
import tvb_data.regionMapping
from os import path

from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.entities.file.simulator.view_model import CortexViewModel, SimulatorAdapterModel
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.services.project_service import initialize_storage
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class TestSimulatorAdapter(TransactionalTestCase):
    """
    Basic testing that Simulator is still working from UI.
    """
    CONNECTIVITY_NODES = 76

    def transactional_setup_method(self):
        """
        Reset the database before each test.
        """
        initialize_storage()

        algorithm = dao.get_algorithm_by_module(IntrospectionRegistry.SIMULATOR_MODULE,
                                                IntrospectionRegistry.SIMULATOR_CLASS)
        self.simulator_adapter = ABCAdapter.build_adapter(algorithm)
        self.test_user = TestFactory.create_user("Simulator_Adapter_User")
        self.test_project = TestFactory.create_project(self.test_user, "Simulator_Adapter_Project")

    def test_happy_flow_launch(self, connectivity_index_factory, operation_factory):
        """
        Test that launching a simulation from UI works.
        """
        model = SimulatorAdapterModel()
        model.connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES).gid
        model.simulation_length = 32

        TestFactory.launch_synchronously(self.test_user.id, self.test_project, self.simulator_adapter, model)
        sim_result = dao.get_generic_entity(TimeSeriesRegionIndex, 'TimeSeriesRegion', 'time_series_type')[0]
        assert (sim_result.data_length_1d, sim_result.data_length_2d, sim_result.data_length_3d,
                sim_result.data_length_4d) == (32, 1, self.CONNECTIVITY_NODES, 1)

    def _estimate_hdd(self, model):
        """ Private method, to return HDD estimation for a given a model"""
        self.simulator_adapter.configure(model)
        return self.simulator_adapter.get_required_disk_size(model)

    def test_estimate_hdd(self, connectivity_index_factory):
        """
        Test that occupied HDD estimation for simulation results considers simulation length.
        """
        model = SimulatorAdapterModel()
        model.connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES).gid
        estimate1 = self._estimate_hdd(model)
        assert estimate1 > 1

        # Change simulation length and monitor period, we expect a direct proportional increase in estimated HDD
        factor = 3
        model.simulation_length = float(model.simulation_length) * factor
        period = float(model.monitors[0].period)
        model.monitors[0].period = period / factor
        estimate2 = self._estimate_hdd(model)
        assert estimate1 == estimate2 // factor // factor

        # Change number of nodes in connectivity. Expect HDD estimation increase.
        model.connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES * factor).gid
        estimate3 = self._estimate_hdd(model)
        assert estimate2 == estimate3 / factor

    def test_estimate_execution_time(self, connectivity_index_factory):
        """
        Test that get_execution_time_approximation considers the correct params
        """
        model = SimulatorAdapterModel()
        model.connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES).gid

        self.simulator_adapter.configure(model)
        estimation1 = self.simulator_adapter.get_execution_time_approximation(model)

        # import surfaceData and region mapping
        cortex_file = path.join(path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        surface = TestFactory.import_surface_zip(self.test_user, self.test_project, cortex_file,
                                                 SurfaceTypesEnum.CORTICAL_SURFACE)
        rm_file = path.join(path.dirname(tvb_data.regionMapping.__file__), 'regionMapping_16k_76.txt')
        region_mapping = TestFactory.import_region_mapping(self.test_user, self.test_project, rm_file, surface.gid,
                                                           model.connectivity.hex)
        local_conn = TestFactory.create_local_connectivity(self.test_user, self.test_project, surface.gid)
        cortex_model = CortexViewModel()
        cortex_model.region_mapping_data = region_mapping.gid
        cortex_model.fk_surface_gid = surface.gid
        cortex_model.local_connectivity = local_conn.gid
        model.surface = cortex_model

        # Estimation when the surface input parameter is set
        self.simulator_adapter.configure(model)
        estimation2 = self.simulator_adapter.get_execution_time_approximation(model)

        assert estimation1 == estimation2 // 500
        model.surface = None

        # Modify integration step and simulation length:
        initial_simulation_length = model.simulation_length
        initial_integration_step = model.integrator.dt

        for factor in (2, 4, 10):
            model.simulation_length = initial_simulation_length * factor
            model.integrator.dt = initial_integration_step / factor
            self.simulator_adapter.configure(model)
            estimation3 = self.simulator_adapter.get_execution_time_approximation(model)

            assert estimation1 == estimation3 // factor // factor

        # Check that no division by zero happens
        model.integrator.dt = 0
        estimation4 = self.simulator_adapter.get_execution_time_approximation(model)
        assert estimation4 > 0

        # even with length zero, still a positive estimation should be returned
        model.simulation_length = 0
        estimation5 = self.simulator_adapter.get_execution_time_approximation(model)
        assert estimation5 > 0
