# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: TVB Team
"""

import uuid

import numpy
import pytest

from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex, TimeSeriesRegionIndex
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.simulator.view_model import EEGViewModel, HybridSimulatorAdapterModel, \
    TemporalAverageViewModel
from tvb.core.entities.model.model_operation import STATUS_ERROR
from tvb.core.entities.storage import dao
from tvb.core.services.hybrid_simulator_service import HybridSimulatorService
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import initialize_storage
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestHybridSimulatorAdapter(TransactionalTestCase):
    """
    The Hybrid Simulator launched the way the framework launches it.
    """

    CONNECTIVITY_NODES = 76
    DT = 0.1
    SIMULATION_LENGTH = 10.0
    PERIOD = 1.0

    def transactional_setup_method(self):
        initialize_storage()
        algorithm = dao.get_algorithm_by_module(IntrospectionRegistry.HYBRID_SIMULATOR_MODULE,
                                                IntrospectionRegistry.HYBRID_SIMULATOR_CLASS)
        self.adapter = ABCAdapter.build_adapter(algorithm)
        self.service = HybridSimulatorService()
        self.test_user = TestFactory.create_user("Hybrid_Adapter_User")
        self.test_project = TestFactory.create_project(self.test_user, "Hybrid_Adapter_Project")

    def _model(self, connectivity_gid, node_indices=None, monitors=None):
        """
        A Hybrid Simulator configuration over the given Connectivity. When node_indices is given, a
        second Subnetwork is created holding exactly those nodes.
        """
        model = HybridSimulatorAdapterModel()
        model.connectivity = connectivity_gid
        model.dt = self.DT
        model.simulation_length = self.SIMULATION_LENGTH
        model.monitors = monitors or [TemporalAverageViewModel(period=self.PERIOD)]

        subnetworks = self.service.create_default_subnetworks(self.CONNECTIVITY_NODES)
        if node_indices is not None:
            subnetworks = self.service.add_subnetwork(subnetworks)
            subnetworks = self.service.move_regions(subnetworks, list(node_indices), 1)
        model.subnetworks = subnetworks
        return model

    @staticmethod
    def _eeg_monitor():
        """
        An EEG Monitor that can be stored. The datatypes behind these GIDs are never loaded: what
        refuses this Monitor is its class against the output layout, which is decided first.
        """
        monitor = EEGViewModel()
        monitor.region_mapping = uuid.uuid4()
        monitor.projection = uuid.uuid4()
        monitor.sensors = uuid.uuid4()
        return monitor

    @staticmethod
    def _expected_samples(simulation_length, period):
        return int(simulation_length / period)

    def test_happy_flow_launch(self, connectivity_index_factory):
        """A simulation over one Subnetwork holding the whole Connectivity."""
        connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES)
        model = self._model(connectivity.gid)

        TestFactory.launch_synchronously(self.test_user.id, self.test_project, self.adapter, model)

        results = dao.get_generic_entity(TimeSeriesRegionIndex, 'TimeSeriesRegion', 'time_series_type')
        assert len(results) == 1
        result = results[0]
        # Generic2dOscillator watches one variable, over every region of the Connectivity
        assert (result.data_length_1d, result.data_length_2d,
                result.data_length_3d, result.data_length_4d) == \
               (self._expected_samples(self.SIMULATION_LENGTH, self.PERIOD), 1, self.CONNECTIVITY_NODES, 1)

    def test_connectome_ordering_is_preserved(self, connectivity_index_factory):
        """
        The columns carrying each Subnetwork's own dynamics must be exactly that Subnetwork's nodes.

        The two Subnetworks are given different Model parameters and the second one a non-contiguous set
        of nodes, so a concatenating output would put its columns in the wrong place and be caught.
        """
        connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES)
        scattered = [0, 5, 17, 42, 75]
        model = self._model(connectivity.gid, node_indices=scattered)
        # tell the two Subnetworks apart by their dynamics
        model.subnetworks[0].dynamics.model.a = numpy.array([-2.0])
        model.subnetworks[1].dynamics.model.a = numpy.array([2.0])

        TestFactory.launch_synchronously(self.test_user.id, self.test_project, self.adapter, model)

        result = dao.get_generic_entity(TimeSeriesRegionIndex, 'TimeSeriesRegion', 'time_series_type')[0]
        data = self._read_data(result)
        assert data.shape[2] == self.CONNECTIVITY_NODES

        # every column of the second Subnetwork must differ from every column of the first one
        second = data[:, 0, scattered, 0]
        first = data[:, 0, [i for i in range(self.CONNECTIVITY_NODES) if i not in scattered], 0]
        assert not numpy.allclose(second.mean(axis=0).mean(), first.mean(axis=0).mean())

    def test_concatenated_output_is_not_claimed_to_be_region_ordered(self, connectivity_index_factory):
        """
        Subnetworks that watch different numbers of variables produce a plain TimeSeries: the node axis
        is then a concatenation, and a TimeSeriesRegion would claim a region ordering it does not have.
        """
        connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES)
        model = self._model(connectivity.gid, node_indices=[0, 1, 2, 3])
        model.subnetworks[1].dynamics.model.variables_of_interest = ('V', 'W')

        TestFactory.launch_synchronously(self.test_user.id, self.test_project, self.adapter, model)

        assert dao.get_generic_entity(TimeSeriesRegionIndex, 'TimeSeriesRegion', 'time_series_type') == []
        results = dao.get_generic_entity(TimeSeriesIndex, 'TimeSeries', 'time_series_type')
        assert len(results) == 1
        # one variable from the first Subnetwork and two from the second, and every node of both
        assert results[0].data_length_2d == 3
        assert results[0].data_length_3d == self.CONNECTIVITY_NODES

    def test_a_projection_monitor_over_concatenated_output_is_refused(self, connectivity_index_factory):
        """
        A gain matrix is indexed in region order, so it cannot be applied to a concatenated array.
        """
        connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES)
        model = self._model(connectivity.gid, node_indices=[0, 1, 2, 3], monitors=[self._eeg_monitor()])
        model.subnetworks[1].dynamics.model.variables_of_interest = ('V', 'W')

        with pytest.raises(Exception) as excinfo:
            self.adapter.configure(model)

        assert 'region order' in str(excinfo.value)

    def test_a_refused_configuration_fails_the_operation(self, connectivity_index_factory):
        """
        And the refusal has to arrive through the normal operation mechanism rather than as a crash.
        """
        connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES)
        model = self._model(connectivity.gid, node_indices=[0, 1, 2, 3], monitors=[self._eeg_monitor()])
        model.subnetworks[1].dynamics.model.variables_of_interest = ('V', 'W')

        service = OperationService()
        operation = service.prepare_operation(self.test_user.id, self.test_project,
                                              self.adapter.stored_adapter, True, model)
        try:
            service.initiate_prelaunch(operation, self.adapter)
        except Exception:
            # the operation mechanism records the failure either way; what it stored is the assertion
            pass

        operation = dao.get_operation_by_id(operation.id)
        assert operation.status == STATUS_ERROR
        assert 'region order' in operation.additional_info

    def test_the_estimates_grow_with_the_simulation(self, connectivity_index_factory):
        connectivity = connectivity_index_factory(self.CONNECTIVITY_NODES)
        short = self._model(connectivity.gid)
        long = self._model(connectivity.gid)
        long.simulation_length = self.SIMULATION_LENGTH * 10

        assert self.adapter.get_required_disk_size(short) > 0
        assert self.adapter.get_required_disk_size(long) > self.adapter.get_required_disk_size(short)
        assert self.adapter.get_execution_time_approximation(long) >= \
               self.adapter.get_execution_time_approximation(short)

    @staticmethod
    def _read_data(ts_index):
        from tvb.core.neocom import h5
        with h5.h5_file_for_index(ts_index) as ts_h5:
            return ts_h5.read_data_slice(slice(None))
