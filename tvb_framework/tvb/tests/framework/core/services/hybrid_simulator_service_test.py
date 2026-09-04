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

import json

import numpy

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

    def test_copy_subnetworks_is_independent_of_the_original(self):
        copied = self.service.copy_subnetworks(self.subnetworks)

        assert self.service.same_grouping(copied, self.subnetworks)
        assert copied[0] is not self.subnetworks[0]

        # the grouping operations change the view models in place, which must not reach the original
        copied = self.service.rename_subnetwork(copied, 0, 'Cortex')
        copied = self.service.move_regions(self.service.add_subnetwork(copied), [0, 1], 1)

        assert self.subnetworks[0].name == 'Subnetwork A'
        assert list(self.subnetworks[0].node_indices) == list(range(self.NUMBER_OF_REGIONS))
        self._assert_valid_partition(self.subnetworks)
        self._assert_valid_partition(copied)

    def test_same_grouping_compares_names_and_assigned_nodes(self):
        copied = self.service.copy_subnetworks(self.subnetworks)
        assert self.service.same_grouping(copied, self.subnetworks)

        renamed = self.service.rename_subnetwork(self.service.copy_subnetworks(self.subnetworks), 0, 'Cortex')
        assert not self.service.same_grouping(renamed, self.subnetworks)

        regrouped = self.service.move_regions(
            self.service.add_subnetwork(self.service.copy_subnetworks(self.subnetworks)), [0], 1)
        assert not self.service.same_grouping(regrouped, self.subnetworks)

    def test_same_grouping_of_nothing(self):
        assert self.service.same_grouping([], None)
        assert not self.service.same_grouping(self.subnetworks, [])

    # ---------------------------------------------------------------- Subnetwork dynamics

    def test_every_subnetwork_gets_its_own_identifier_and_dynamics(self):
        subnetworks = self.service.add_subnetwork(self.subnetworks)

        assert subnetworks[0].id != subnetworks[1].id
        # a trait default would be one shared instance, and the parameter forms edit these in place
        assert subnetworks[0].dynamics is not subnetworks[1].dynamics
        assert subnetworks[0].model is not subnetworks[1].model
        assert subnetworks[0].integrator is not subnetworks[1].integrator

    def test_copy_keeps_the_identifier_and_detaches_the_dynamics(self):
        copied = self.service.copy_subnetworks(self.subnetworks)

        assert copied[0].id == self.subnetworks[0].id
        assert copied[0].dynamics is not self.subnetworks[0].dynamics

        copied[0].model.a = numpy.array([9.0])
        assert list(self.subnetworks[0].model.a) != [9.0]

    def test_dynamics_draft_is_seeded_from_the_saved_dynamics(self):
        draft = self.service.prepare_dynamics_draft(self.subnetworks, None)

        assert set(draft.keys()) == {self.subnetworks[0].id}
        assert self.service.same_dynamics(self.subnetworks, draft)
        # and it is a copy, so editing it leaves the saved configuration alone
        assert draft[self.subnetworks[0].id] is not self.subnetworks[0].dynamics

    def test_dynamics_draft_reports_an_edit(self):
        draft = self.service.prepare_dynamics_draft(self.subnetworks, None)
        draft[self.subnetworks[0].id].model.a = numpy.array([9.0])

        assert not self.service.same_dynamics(self.subnetworks, draft)

    def test_dynamics_draft_drops_entries_of_subnetworks_that_are_gone(self):
        subnetworks = self.service.add_subnetwork(self.subnetworks)
        draft = self.service.prepare_dynamics_draft(subnetworks, None)
        removed_id = subnetworks[1].id

        remaining = self.service.remove_subnetwork(subnetworks, 1)
        draft = self.service.prepare_dynamics_draft(remaining, draft)

        assert removed_id not in draft
        assert set(draft.keys()) == {remaining[0].id}

    def test_dynamics_draft_survives_a_rename(self):
        draft = self.service.prepare_dynamics_draft(self.subnetworks, None)
        draft[self.subnetworks[0].id].model.a = numpy.array([9.0])

        renamed = self.service.rename_subnetwork(self.subnetworks, 0, 'Cortex')
        draft = self.service.prepare_dynamics_draft(renamed, draft)

        assert list(draft[renamed[0].id].model.a) == [9.0]

    def test_store_dynamics_writes_the_draft_onto_the_subnetworks(self):
        draft = self.service.prepare_dynamics_draft(self.subnetworks, None)
        draft[self.subnetworks[0].id].model.a = numpy.array([9.0])

        self.service.store_dynamics(self.subnetworks, draft)

        assert list(self.subnetworks[0].model.a) == [9.0]
        assert self.service.same_dynamics(self.subnetworks, draft)
        # stored as a copy, so continuing to edit the draft does not change what was saved
        draft[self.subnetworks[0].id].model.a = numpy.array([3.0])
        assert list(self.subnetworks[0].model.a) == [9.0]

    def test_shared_dt_reaches_every_integrator(self):
        subnetworks = self.service.add_subnetwork(self.subnetworks)
        draft = self.service.prepare_dynamics_draft(subnetworks, None)

        self.service.apply_shared_dt(draft, 0.5)

        assert [dynamics.integrator.dt for dynamics in draft.values()] == [0.5, 0.5]

    def test_find_subnetwork_refuses_an_unknown_identifier(self):
        with pytest.raises(HybridSubnetworkException):
            self.service.find_subnetwork(self.subnetworks, 'not-an-identifier')

        assert self.service.find_subnetwork(self.subnetworks, self.subnetworks[0].id) is self.subnetworks[0]

    # ---------------------------------------------------------------- Model parameter shapes

    def test_model_parameters_may_be_shared_or_one_per_owned_node(self):
        self.service.validate_model_parameters(self.subnetworks[0])

        self.subnetworks[0].model.a = numpy.array([1.0] * self.NUMBER_OF_REGIONS)
        self.service.validate_model_parameters(self.subnetworks[0])

    def test_model_parameters_sized_for_another_node_set_are_refused(self):
        subnetworks = self.service.move_regions(
            self.service.add_subnetwork(self.subnetworks), [0, 1], 1)
        # the second Subnetwork owns 2 of the 8 regions, so a value per Connectivity node cannot apply
        subnetworks[1].model.a = numpy.array([1.0] * self.NUMBER_OF_REGIONS)

        with pytest.raises(HybridSubnetworkException) as excep:
            self.service.validate_model_parameters(subnetworks[1])

        message = str(excep.value)
        assert subnetworks[1].name in message
        assert "'a'" in message
        assert '2' in message

    # ---------------------------------------------------------------- tvb_library naming

    def test_names_are_sanitized_into_identifiers(self):
        subnetworks = self.service.rename_subnetwork(self.subnetworks, 0, 'Subnetwork A')
        identifiers = self.service.to_identifiers(subnetworks)

        # NetworkSet builds a namedtuple out of these, so each one has to be a valid identifier
        assert identifiers[subnetworks[0].id] == 'Subnetwork_A'
        assert identifiers[subnetworks[0].id].isidentifier()

    def test_names_sanitizing_to_the_same_identifier_are_disambiguated(self):
        subnetworks = self.service.rename_subnetwork(self.subnetworks, 0, 'Sub A')
        subnetworks = self.service.add_subnetwork(subnetworks)
        subnetworks = self.service.rename_subnetwork(subnetworks, 1, 'Sub-A')

        identifiers = self.service.to_identifiers(subnetworks)

        assert len(set(identifiers.values())) == 2
        for identifier in identifiers.values():
            assert identifier.isidentifier()

    def test_names_that_are_not_identifiers_at_all(self):
        subnetworks = self.service.rename_subnetwork(self.subnetworks, 0, '2nd network!')
        identifiers = self.service.to_identifiers(subnetworks)

        identifier = identifiers[subnetworks[0].id]
        assert identifier.isidentifier(), identifier
        assert not identifier[0].isdigit()

    # ---------------------------------------------------------------- Set up region Model

    class _FakeDynamic(object):
        """A saved Dynamic, as far as this service is concerned: a model class and its parameters."""

        def __init__(self, dynamic_id, name, model_class, parameters):
            self.id = dynamic_id
            self.name = name
            self.model_class = model_class
            # a Dynamic stores its parameters as a JSON list of name/value pairs
            self.model_parameters = json.dumps([[key, value] for key, value in parameters.items()])

    def _dynamics(self):
        fast = self._FakeDynamic(1, 'fast', 'Generic2dOscillator', {'a': -2.0, 'tau': 1.0})
        slow = self._FakeDynamic(2, 'slow', 'Generic2dOscillator', {'a': -4.0, 'tau': 1.0})
        other = self._FakeDynamic(3, 'other', 'Kuramoto', {'omega': 1.0})
        return fast, slow, other

    def test_only_dynamics_of_the_configured_model_class_are_offered(self):
        fast, slow, other = self._dynamics()
        model = self.subnetworks[0].model

        offered = self.service.dynamics_for_model([fast, slow, other], model)

        assert [dynamic.name for dynamic in offered] == ['fast', 'slow']

    def test_no_dynamics_are_offered_without_a_model(self):
        fast, _, _ = self._dynamics()
        assert self.service.dynamics_for_model([fast], None) == []

    def test_applying_dynamics_writes_one_value_per_node(self):
        fast, slow, _ = self._dynamics()
        dynamics_by_id = {fast.id: fast, slow.id: slow}
        node_indices = [0, 1, 2]
        assignment = {0: slow.id, 1: fast.id, 2: fast.id}

        model = self.service.apply_dynamics_to_model(
            self.subnetworks[0].model, node_indices, assignment, dynamics_by_id)

        assert list(model.a) == [-4.0, -2.0, -2.0]
        # a parameter every configuration agrees on contracts back to a single shared value
        assert list(model.tau) == [1.0]

    def test_applying_dynamics_refuses_an_unconfigured_region(self):
        fast, _, _ = self._dynamics()
        assignment = {0: fast.id}

        with pytest.raises(HybridSubnetworkException) as excep:
            self.service.apply_dynamics_to_model(
                self.subnetworks[0].model, [0, 1, 2], assignment, {fast.id: fast})

        assert '2' in str(excep.value)

    def test_applying_dynamics_ignores_parameters_the_model_does_not_declare(self):
        stray = self._FakeDynamic(9, 'stray', 'Generic2dOscillator', {'a': -2.0, 'not_a_parameter': 3.0})

        model = self.service.apply_dynamics_to_model(
            self.subnetworks[0].model, [0], {0: stray.id}, {stray.id: stray})

        assert list(model.a) == [-2.0]
        assert not hasattr(model, 'not_a_parameter')

    def test_placing_a_dynamic_leaves_the_other_regions_alone(self):
        assignment = self.service.place_dynamic_on_regions({0: 5}, [1, 2], 7)

        assert assignment == {0: 5, 1: 7, 2: 7}

    def test_assignment_is_restricted_to_the_regions_still_owned(self):
        assignment = self.service.restrict_assignment({0: 1, 1: 1, 5: 2}, [1, 5])

        assert assignment == {1: 1, 5: 2}

    def test_unassigned_count(self):
        assert self.service.unassigned_count([0, 1, 2], {0: 1}) == 2
        assert self.service.unassigned_count([0, 1], {0: 1, 1: 2}) == 0

    def test_region_model_rows_describe_every_owned_region(self):
        fast, _, _ = self._dynamics()

        rows = self.service.region_model_rows([0, 2], ['lOFC', 'rOFC', 'lPCUN'], {0: fast.id},
                                              {fast.id: fast})

        assert rows[0] == {'index': 0, 'label': 'lOFC', 'dynamic_id': fast.id, 'dynamic_name': 'fast'}
        assert rows[1] == {'index': 2, 'label': 'lPCUN', 'dynamic_id': None, 'dynamic_name': ''}
