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

import pytest

from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel, HybridSubnetworkViewModel
from tvb.core.services.hybrid_simulator_service import HybridSimulatorService, HybridSubnetworkException


class TestHybridSimulatorService(object):
    """
    Focused tests for the Subnetwork grouping logic, without any Connectivity storage involved.
    """

    NUMBER_OF_REGIONS = 8

    def setup_method(self):
        self.service = HybridSimulatorService()
        self.subnetworks = self.service.create_default_subnetworks(self.NUMBER_OF_REGIONS)

    def _assert_valid_partition(self, subnetworks):
        assigned = []
        for subnetwork in subnetworks:
            assigned.extend(subnetwork.node_indices)

        assert sorted(assigned) == list(range(self.NUMBER_OF_REGIONS))
        assert len(assigned) == len(set(assigned))

    def test_default_configuration_holds_every_region(self):
        assert len(self.subnetworks) == 1
        assert self.subnetworks[0].name == 'Subnetwork A'
        assert list(self.subnetworks[0].node_indices) == list(range(self.NUMBER_OF_REGIONS))
        self._assert_valid_partition(self.subnetworks)

    def test_default_names_do_not_repeat(self):
        names = set()
        for _ in range(30):
            self.subnetworks = self.service.add_subnetwork(self.subnetworks)
            names.add(self.subnetworks[-1].name)

        assert len(names) == 30
        assert self.subnetworks[1].name == 'Subnetwork B'
        assert self.subnetworks[26].name == 'Subnetwork AA'

    def test_add_subnetwork_creates_an_empty_one(self):
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)

        assert len(self.subnetworks) == 2
        assert list(self.subnetworks[1].node_indices) == []
        self._assert_valid_partition(self.subnetworks)

    def test_rename_subnetwork(self):
        self.subnetworks = self.service.rename_subnetwork(self.subnetworks, 0, '  Thalamus ')
        assert self.subnetworks[0].name == 'Thalamus'

    @pytest.mark.parametrize('name', ['', '   ', None])
    def test_rename_subnetwork_refuses_empty_name(self, name):
        with pytest.raises(HybridSubnetworkException):
            self.service.rename_subnetwork(self.subnetworks, 0, name)

    def test_rename_subnetwork_refuses_duplicated_name(self):
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)

        with pytest.raises(HybridSubnetworkException):
            self.service.rename_subnetwork(self.subnetworks, 1, 'Subnetwork A')

    def test_rename_subnetwork_keeps_its_own_name(self):
        self.subnetworks = self.service.rename_subnetwork(self.subnetworks, 0, 'Subnetwork A')
        assert self.subnetworks[0].name == 'Subnetwork A'

    def test_move_regions_between_subnetworks(self):
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)
        self.subnetworks = self.service.move_regions(self.subnetworks, [5, 1, 3], 1)

        assert list(self.subnetworks[1].node_indices) == [1, 3, 5]
        assert list(self.subnetworks[0].node_indices) == [0, 2, 4, 6, 7]
        self._assert_valid_partition(self.subnetworks)

    def test_moved_regions_belong_to_a_single_subnetwork(self):
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)

        self.subnetworks = self.service.move_regions(self.subnetworks, [0, 1, 2], 1)
        self.subnetworks = self.service.move_regions(self.subnetworks, [1, 2], 2)

        assert list(self.subnetworks[1].node_indices) == [0]
        assert list(self.subnetworks[2].node_indices) == [1, 2]
        self._assert_valid_partition(self.subnetworks)

    def test_move_regions_refuses_unknown_input(self):
        with pytest.raises(HybridSubnetworkException):
            self.service.move_regions(self.subnetworks, [], 0)

        with pytest.raises(HybridSubnetworkException):
            self.service.move_regions(self.subnetworks, [self.NUMBER_OF_REGIONS], 0)

        with pytest.raises(HybridSubnetworkException):
            self.service.move_regions(self.subnetworks, ['not-a-node'], 0)

        with pytest.raises(HybridSubnetworkException):
            self.service.move_regions(self.subnetworks, [0], 3)

        self._assert_valid_partition(self.subnetworks)

    def test_remove_subnetwork_moves_its_regions_to_the_first_one(self):
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)
        self.subnetworks = self.service.move_regions(self.subnetworks, [2, 6], 1)
        self.subnetworks = self.service.remove_subnetwork(self.subnetworks, 1)

        assert len(self.subnetworks) == 1
        self._assert_valid_partition(self.subnetworks)

    def test_remove_first_subnetwork_moves_its_regions_to_the_next_one(self):
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)
        self.subnetworks = self.service.move_regions(self.subnetworks, [2, 6], 1)
        self.subnetworks = self.service.remove_subnetwork(self.subnetworks, 0)

        assert len(self.subnetworks) == 1
        assert self.subnetworks[0].name == 'Subnetwork B'
        self._assert_valid_partition(self.subnetworks)

    def test_remove_refuses_to_leave_no_subnetwork(self):
        with pytest.raises(HybridSubnetworkException):
            self.service.remove_subnetwork(self.subnetworks, 0)

        self._assert_valid_partition(self.subnetworks)

    def test_discard_empty_subnetworks(self):
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)
        self.subnetworks = self.service.add_subnetwork(self.subnetworks)
        self.subnetworks = self.service.move_regions(self.subnetworks, [1, 2], 1)

        remaining = self.service.discard_empty_subnetworks(self.subnetworks)

        assert [subnetwork.name for subnetwork in remaining] == ['Subnetwork A', 'Subnetwork B']
        self._assert_valid_partition(remaining)

    def test_discard_empty_subnetworks_keeps_a_populated_configuration_untouched(self):
        remaining = self.service.discard_empty_subnetworks(self.subnetworks)

        assert remaining == self.subnetworks
        self._assert_valid_partition(remaining)

    def test_discard_empty_subnetworks_always_keeps_one(self):
        empty_only = [HybridSubnetworkViewModel(name='Subnetwork A', node_indices=[])]

        remaining = self.service.discard_empty_subnetworks(empty_only)

        assert len(remaining) == 1
        assert remaining[0].name == 'Subnetwork A'

    def test_remove_refuses_an_unknown_position(self):
        with pytest.raises(HybridSubnetworkException):
            self.service.remove_subnetwork(self.subnetworks, 4)

    def test_prepare_subnetworks_keeps_a_valid_configuration(self):
        hybrid_simulator = HybridSimulatorAdapterModel()
        hybrid_simulator.subnetworks = self.service.move_regions(
            self.service.add_subnetwork(self.subnetworks), [0, 1], 1)

        prepared = self.service.prepare_subnetworks(hybrid_simulator, self.NUMBER_OF_REGIONS)

        assert len(prepared) == 2
        assert list(prepared[1].node_indices) == [0, 1]

    def test_prepare_subnetworks_resets_an_inconsistent_configuration(self):
        hybrid_simulator = HybridSimulatorAdapterModel()
        hybrid_simulator.subnetworks = [HybridSubnetworkViewModel(name='Stale', node_indices=[0, 1])]

        prepared = self.service.prepare_subnetworks(hybrid_simulator, self.NUMBER_OF_REGIONS)

        assert len(prepared) == 1
        assert prepared[0].name == 'Subnetwork A'
        assert list(prepared[0].node_indices) == list(range(self.NUMBER_OF_REGIONS))
        assert hybrid_simulator.subnetworks is prepared

    def test_prepare_subnetworks_creates_the_default_configuration(self):
        hybrid_simulator = HybridSimulatorAdapterModel()

        prepared = self.service.prepare_subnetworks(hybrid_simulator, self.NUMBER_OF_REGIONS)

        assert len(prepared) == 1
        assert list(prepared[0].node_indices) == list(range(self.NUMBER_OF_REGIONS))
