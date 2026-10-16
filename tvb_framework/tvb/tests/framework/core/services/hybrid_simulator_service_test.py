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

from tvb.core.entities.file.simulator.view_model import BoldViewModel, HybridSimulatorAdapterModel, \
    HybridSubnetworkViewModel, RawViewModel, TemporalAverageViewModel
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

    def test_region_labels_read_from_storage_are_strings(self, tmpdir):
        from unittest.mock import patch
        from tvb.adapters.datatypes.h5.connectivity_h5 import ConnectivityH5
        from tvb.datatypes.connectivity import Connectivity

        connectivity = Connectivity(region_labels=numpy.array(['lA1', 'rV2']), weights=numpy.ones((2, 2)),
                                    tract_lengths=numpy.ones((2, 2)), centres=numpy.zeros((2, 3)))
        path = str(tmpdir.join('Connectivity.h5'))
        with ConnectivityH5(path) as conn_h5:
            conn_h5.store(connectivity)

        # H5 hands the labels back as bytes
        with patch('tvb.core.neocom.h5.h5_file_for_gid', lambda gid: ConnectivityH5(path)):
            labels = self.service.get_region_labels(connectivity.gid)

        assert labels == ['lA1', 'rV2']
        assert all(type(label) is str for label in labels)

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

    # ---------------------------------------------------------------- tvb_library translation

    def _connectivity(self):
        """
        A deterministic Connectivity over this test's 8 nodes: two blocks, connected one way only.
        Weights are indexed (target, source), which is TVB's own orientation.
        """
        from tvb.datatypes.connectivity import Connectivity

        n = self.NUMBER_OF_REGIONS
        weights = numpy.zeros((n, n))
        weights[:4, :4] = 1.0        # inside the first block
        weights[4:, 4:] = 2.0        # inside the second block
        weights[4:, :4] = 0.5        # first block -> second block, and nothing back
        lengths = numpy.full((n, n), 10.0)

        return Connectivity(weights=weights, tract_lengths=lengths,
                            region_labels=numpy.array(['r%d' % index for index in range(n)]),
                            centres=numpy.zeros((n, 3)), number_of_regions=n)

    def _build_all(self, connectivity, subnetworks, dt):
        """
        Build the NetworkSet with every generated Projection added to the simulation: no Projection is
        until the user adds it, and what these tests look at is what the generation produces.
        """
        projections, _ = self.service.reconcile_projections(connectivity, subnetworks, None)
        for projection in projections:
            projection.included = True
        return self.service.build_network_set(connectivity, subnetworks, dt, projections)

    def _two_blocks(self):
        """Nodes 0-3 in the first Subnetwork, 4-7 in the second."""
        subnetworks = self.service.add_subnetwork(self.subnetworks)
        return self.service.move_regions(subnetworks, [4, 5, 6, 7], 1)

    def test_library_subnetworks_carry_the_configuration(self):
        subnetworks = self._two_blocks()

        built = self.service.build_library_subnetworks(subnetworks, 0.25)

        assert [subnet.nnodes for subnet in built] == [4, 4]
        assert [list(subnet.node_indices) for subnet in built] == [[0, 1, 2, 3], [4, 5, 6, 7]]
        # the shared dt reaches every scheme, which is what stops Simulator.validate_dts from refusing
        assert [subnet.scheme.dt for subnet in built] == [0.25, 0.25]
        # names have to be identifiers: NetworkSet builds a namedtuple out of them
        for subnet in built:
            assert subnet.name.isidentifier()

    def test_library_subnetworks_do_not_share_the_configured_objects(self):
        subnetworks = self._two_blocks()

        built = self.service.build_library_subnetworks(subnetworks, 0.1)

        # configure() mutates what it is given, and the configuration objects are the ones the forms
        # keep editing, so the built Subnetworks must hold copies
        assert built[0].model is not subnetworks[0].dynamics.model
        assert built[0].scheme is not subnetworks[0].dynamics.integrator
        assert built[0].model is not built[1].model

    def test_intra_projections_hold_this_subnetworks_own_block(self):
        subnetworks = self._two_blocks()
        connectivity = self._connectivity()

        network_set = self._build_all(connectivity, subnetworks, 0.1)

        first, second = network_set.subnets
        assert len(first.projections) == 1
        intra = first.projections[0]
        assert intra.weights.shape == (4, 4)
        assert numpy.allclose(intra.weights.toarray(), 1.0)
        assert numpy.allclose(intra.lengths.toarray(), 10.0)
        # and the other Subnetwork's own block, which holds a different weight
        assert numpy.allclose(second.projections[0].weights.toarray(), 2.0)

    def test_inter_projections_are_generated_for_connected_pairs_only(self):
        subnetworks = self._two_blocks()
        connectivity = self._connectivity()

        network_set = self._build_all(connectivity, subnetworks, 0.1)

        assert len(network_set.projections) == 1
        projection = network_set.projections[0]
        # the Connectivity connects the first block to the second and nothing back
        assert projection.source is network_set.subnets[0]
        assert projection.target is network_set.subnets[1]
        assert projection.weights.shape == (4, 4)
        assert numpy.allclose(projection.weights.toarray(), 0.5)

        # the pair left out is reported rather than quietly dropped
        unconnected = self.service.unconnected_pairs(network_set)
        assert len(unconnected) == 1
        assert unconnected[0]['source'] == network_set.subnets[1].name
        assert unconnected[0]['target'] == network_set.subnets[0].name

    def test_coupling_variables_are_left_at_safe_defaults(self):
        subnetworks = self._two_blocks()
        connectivity = self._connectivity()

        network_set = self._build_all(connectivity, subnetworks, 0.1)
        model = network_set.subnets[0].model

        projection = network_set.projections[0]
        # source_cvar indexes the history buffer, so it is a state variable index
        assert list(numpy.atleast_1d(projection.source_cvar)) == [int(model.cvar[0])]
        # target_cvar indexes the coupling array, so it is a slot in the target model's cvar list
        assert list(numpy.atleast_1d(projection.target_cvar)) == [0]

    def test_the_network_set_names_its_states_after_the_subnetworks(self):
        subnetworks = self._two_blocks()
        connectivity = self._connectivity()

        network_set = self._build_all(connectivity, subnetworks, 0.1)

        # this is what would break on a name that is not a valid Python identifier
        assert network_set.States._fields == tuple(subnet.name for subnet in network_set.subnets)

    def test_a_single_subnetwork_gets_an_intra_projection_and_no_inter_one(self):
        connectivity = self._connectivity()

        network_set = self._build_all(connectivity, self.subnetworks, 0.1)

        assert len(network_set.subnets) == 1
        assert len(network_set.subnets[0].projections) == 1
        assert network_set.subnets[0].projections[0].weights.shape == (8, 8)
        assert network_set.projections == []

    def test_the_generated_network_set_is_described_for_the_wizard(self):
        subnetworks = self._two_blocks()
        connectivity = self._connectivity()

        network_set = self._build_all(connectivity, subnetworks, 0.1)
        rows = self.service.describe_network_set(network_set)

        kinds = [row['kind'] for row in rows]
        assert kinds == ['Intra', 'Intra', 'Inter']
        assert rows[0]['shape'] == '4 x 4'
        assert rows[0]['connections'] == 16
        assert rows[2]['source'] == network_set.subnets[0].name
        assert rows[2]['target'] == network_set.subnets[1].name

    def test_regrouping_changes_what_is_generated(self):
        connectivity = self._connectivity()
        subnetworks = self._two_blocks()
        # move one node across, so the blocks are 3 and 5 nodes wide
        subnetworks = self.service.move_regions(subnetworks, [3], 1)

        network_set = self._build_all(connectivity, subnetworks, 0.1)

        assert [subnet.nnodes for subnet in network_set.subnets] == [3, 5]
        assert network_set.subnets[0].projections[0].weights.shape == (3, 3)
        assert network_set.projections[0].weights.shape == (5, 3)

    # ---------------------------------------------------------------- Global configuration

    def _network_set(self, dt=0.1):
        return self._build_all(self._connectivity(), self._two_blocks(), dt)

    def test_the_configuration_reaches_the_hybrid_simulator(self):
        network_set = self._network_set()

        simulator = self.service.build_hybrid_simulator(
            network_set, [TemporalAverageViewModel()], 500.0)

        assert simulator.nets is network_set
        assert simulator.simulation_length == 500.0
        assert len(simulator.monitors) == 1
        # the library default, the numba one accepts only a fixed set of Models and Integrators
        assert simulator.backend == 'python'

    def test_monitors_are_configured_from_the_shared_dt(self):
        monitor = TemporalAverageViewModel()
        monitor.period = 1.0

        simulator = self.service.build_hybrid_simulator(self._network_set(dt=0.25), [monitor], 100.0)

        built = simulator.monitors[0]
        assert built.dt == 0.25
        # what the Monitor samples on: one sample every period / dt integration steps
        assert built.istep == 4

    def test_the_built_monitors_do_not_share_the_configured_ones(self):
        monitor = TemporalAverageViewModel()
        monitor.period = 1.0

        simulator = self.service.build_hybrid_simulator(self._network_set(), [monitor], 100.0)
        # building configures every Monitor and allocates its buffers, and this is the object the forms
        # keep editing
        monitor.period = 50.0

        assert simulator.monitors[0] is not monitor
        assert simulator.monitors[0].period == 1.0

    def test_simulation_length_is_respected(self):
        monitor = TemporalAverageViewModel()
        monitor.period = 1.0

        simulator = self.service.build_hybrid_simulator(self._network_set(dt=0.1), [monitor], 10.0)
        ((times, data),) = simulator.run()

        # 10 ms sampled every 1 ms
        assert len(times) == 10
        assert data.shape[0] == 10

    def test_a_sampling_period_below_the_step_size_is_refused(self):
        monitor = TemporalAverageViewModel()
        monitor.period = 0.05

        with pytest.raises(HybridSubnetworkException) as excinfo:
            self.service.validate_monitors([monitor], 100.0, 0.1)

        # rounding that period to integration steps gives zero, which the Monitor then divides by
        assert 'TemporalAverage' in str(excinfo.value)
        assert '0.1' in str(excinfo.value)

    def test_a_sampling_period_longer_than_the_simulation_is_refused(self):
        monitor = BoldViewModel()

        with pytest.raises(HybridSubnetworkException) as excinfo:
            self.service.validate_monitors([monitor], 100.0, 0.1)

        assert 'Bold' in str(excinfo.value)
        assert 'record nothing' in str(excinfo.value)

    def test_the_length_rule_waits_until_the_length_is_known(self):
        # the Monitors are configured before the simulation length is asked, so only the step size
        # rule applies to them then
        self.service.validate_monitors([BoldViewModel()], None, 0.1)

        monitor = TemporalAverageViewModel()
        monitor.period = 0.05
        with pytest.raises(HybridSubnetworkException):
            self.service.validate_monitors([monitor], None, 0.1)

    def test_the_raw_monitor_is_not_judged_on_its_period(self):
        # Raw records every integration step and documents its sampling period as ignored, so its
        # default of zero is not a period below dt
        self.service.validate_monitors([RawViewModel()], 100.0, 0.1)

    def test_output_is_connectome_ordered_when_the_subnetworks_agree(self):
        subnetworks = self._two_blocks()

        layout = self.service.output_layout(subnetworks)

        assert layout['is_merged'] is True
        assert layout['variables'] == 1
        # one column per Connectivity region, at its own position
        assert layout['nodes'] == self.NUMBER_OF_REGIONS

    def test_output_is_concatenated_when_the_subnetworks_disagree(self):
        subnetworks = self._two_blocks()
        subnetworks[1].dynamics.model.variables_of_interest = ('V', 'W')

        layout = self.service.output_layout(subnetworks)

        assert layout['is_merged'] is False
        # the variables stacked and the nodes concatenated, Subnetwork after Subnetwork
        assert layout['variables'] == 3
        assert layout['nodes'] == self.NUMBER_OF_REGIONS
        assert [row['name'] for row in layout['rows']] == [subnetworks[0].name, subnetworks[1].name]
        assert layout['rows'][1]['variables'] == ['V', 'W']

    def _hand_built_network_set(self, connectivity, dt=0.1):
        """
        The same two-Subnetwork network, written the way the simulate_hybrid_* demos write it: the
        Connectivity blocks sliced by hand into sparse matrices, and the projections constructed
        directly rather than through the service.

        Every projection couples through Linear with the classic Simulator's defaults, at the
        Connectivity's conduction speed, which is what a Projection nobody configured gets.
        """
        import scipy.sparse as sp
        from tvb.simulator.coupling import Linear as ClassicLinear
        from tvb.simulator.hybrid import IntraProjection, InterProjection, NetworkSet, Subnetwork
        from tvb.simulator.hybrid.coupling import Linear

        def classic_linear():
            return Linear(a=numpy.array(ClassicLinear.a.default), b=numpy.array(ClassicLinear.b.default))

        speed = float(numpy.mean(connectivity.speed))
        from tvb.simulator.integrators import HeunDeterministic
        from tvb.simulator.models import Generic2dOscillator

        first_nodes = [0, 1, 2, 3]
        second_nodes = [4, 5, 6, 7]

        def block(targets, sources):
            # TVB indexes weights and tract lengths as (target, source)
            return (sp.csr_matrix(connectivity.weights[numpy.ix_(targets, sources)]),
                    sp.csr_matrix(connectivity.tract_lengths[numpy.ix_(targets, sources)]))

        subnets = []
        for name, nodes in [('first', first_nodes), ('second', second_nodes)]:
            model = Generic2dOscillator()
            model.configure()
            subnet = Subnetwork(name=name, model=model, scheme=HeunDeterministic(dt=dt),
                                nnodes=len(nodes), node_indices=numpy.array(nodes))
            weights, lengths = block(nodes, nodes)
            subnet.projections = [IntraProjection(
                source_cvar=numpy.array([int(model.cvar[0])]), target_cvar=numpy.array([0]),
                weights=weights, lengths=lengths, cv=speed, dt=dt, scale=1.0, cfun=classic_linear())]
            subnet.configure()
            subnets.append(subnet)

        # this Connectivity connects the first block to the second one and nothing back
        weights, lengths = block(second_nodes, first_nodes)
        projection = InterProjection(
            source=subnets[0], target=subnets[1],
            source_cvar=numpy.array([int(subnets[0].model.cvar[0])]), target_cvar=numpy.array([0]),
            weights=weights, lengths=lengths, cv=speed, dt=dt, scale=1.0, cfun=classic_linear())

        network_set = NetworkSet(subnets=subnets, projections=[projection])
        network_set.configure()
        return network_set

    def test_the_generated_simulation_matches_one_written_by_hand(self):
        """
        What the wizard produces has to be exactly what someone writing the library API by hand gets.

        This is the comparison against the demos that Phase 4 had to defer: it could not be made until a
        simulation could actually be run. The demos' own weights are random, so both sides are built
        over the same Connectivity instead, and both are started from the same initial conditions.
        """
        from tvb.simulator.hybrid import Simulator as LibrarySimulator
        from tvb.simulator.monitors import TemporalAverage

        connectivity = self._connectivity()
        initial_conditions = [numpy.zeros((2, 4, 1)), numpy.zeros((2, 4, 1))]

        network_set = self._build_all(connectivity, self._two_blocks(), 0.1)
        simulator = self.service.build_hybrid_simulator(
            network_set, [TemporalAverageViewModel(period=1.0)], 10.0)
        ((times, data),) = simulator.run(initial_conditions=initial_conditions)

        by_hand = LibrarySimulator(nets=self._hand_built_network_set(connectivity),
                                   monitors=[TemporalAverage(period=1.0)], simulation_length=10.0)
        by_hand.configure()
        ((hand_times, hand_data),) = by_hand.run(initial_conditions=initial_conditions)

        assert numpy.allclose(times, hand_times)
        assert data.shape == hand_data.shape
        assert numpy.allclose(data, hand_data)

    # ---------------------------------------------------------------- Projection configuration

    def _projections(self, connectivity, subnetworks, projections=None):
        reconciled, _ = self.service.reconcile_projections(connectivity, subnetworks, projections)
        return reconciled

    @staticmethod
    def _by_pair(projections, subnetworks):
        names = {subnetwork.id: subnetwork.name for subnetwork in subnetworks}
        return {(names[projection.source_id], names[projection.target_id]): projection
                for projection in projections}

    def test_every_projection_starts_with_linear_at_the_connectivity_speed(self):
        from tvb.simulator.coupling import Linear as ClassicLinear
        from tvb.simulator.hybrid.coupling import Linear

        connectivity = self._connectivity()
        connectivity.speed = numpy.array([5.0])
        subnetworks = self._two_blocks()

        projections = self._projections(connectivity, subnetworks)

        # one IntraProjection per Subnetwork, and the one connected direction between them, none of
        # them part of the simulation until the user adds it
        assert [(projection.is_intra, projection.included) for projection in projections] == \
               [(True, False), (True, False), (False, False)]
        for projection in projections:
            assert type(projection.coupling) is Linear
            assert list(projection.coupling.a) == list(ClassicLinear.a.default)
            assert list(projection.coupling.b) == list(ClassicLinear.b.default)
            assert projection.cv == 5.0
            assert projection.scale == 1.0
            # the first coupling variable of each Model, by name
            assert projection.source_cvar == ['V'] and projection.target_cvar == ['V']

    def test_nothing_is_built_until_it_is_added(self):
        network_set = self.service.build_network_set(self._connectivity(), self._two_blocks(), 0.1)

        assert network_set.projections == []
        assert [subnet.projections for subnet in network_set.subnets] == [[], []]

    def test_the_defaults_reach_the_built_network_set(self):
        from tvb.simulator.hybrid.coupling import Linear

        connectivity = self._connectivity()
        connectivity.speed = numpy.array([5.0])

        # once added, a Projection nobody configured is built with the defaults
        network_set = self._build_all(connectivity, self._two_blocks(), 0.1)

        built = [projection for subnet in network_set.subnets for projection in subnet.projections]
        built += list(network_set.projections)
        assert len(built) == 3
        for projection in built:
            assert isinstance(projection.cfun, Linear)
            assert projection.cv == 5.0

    def test_edited_settings_reach_the_built_network_set(self):
        from tvb.simulator.hybrid.coupling import Kuramoto
        from tvb.simulator.models import JansenRit

        connectivity = self._connectivity()
        subnetworks = self._two_blocks()
        # JansenRit couples through two variables, y1 and y2
        subnetworks[0].dynamics.model = JansenRit()
        projections = self._projections(connectivity, subnetworks)
        inter = self._by_pair(projections, subnetworks)[('Subnetwork A', 'Subnetwork B')]
        inter.scale = 2.5
        inter.cv = 6.0
        inter.coupling = Kuramoto(a=numpy.array([0.3]))
        inter.source_cvar = ['y2']
        inter.target_cvar = ['V']
        inter.included = True

        network_set = self.service.build_network_set(connectivity, subnetworks, 0.1, projections)

        (built,) = network_set.projections
        assert built.scale == 2.5
        assert built.cv == 6.0
        assert isinstance(built.cfun, Kuramoto) and list(built.cfun.a) == [0.3]
        # a source coupling variable is its state variable index, a target one its coupling slot
        assert list(built.source_cvar) == [2]
        assert list(built.target_cvar) == [0]
        # and the configuration itself is not touched by building from it
        assert inter.source_cvar == ['y2']

    def test_an_excluded_projection_is_not_built(self):
        connectivity = self._connectivity()
        subnetworks = self._two_blocks()
        projections = self._projections(connectivity, subnetworks)
        for projection in projections:
            projection.included = True
        by_pair = self._by_pair(projections, subnetworks)
        by_pair[('Subnetwork A', 'Subnetwork B')].included = False
        by_pair[('Subnetwork B', 'Subnetwork B')].included = False

        network_set = self.service.build_network_set(connectivity, subnetworks, 0.1, projections)

        assert network_set.projections == []
        assert [len(subnet.projections) for subnet in network_set.subnets] == [1, 0]

    def test_reconciling_follows_the_grouping(self):
        connectivity = self._connectivity()
        subnetworks = self._two_blocks()
        projections = self._projections(connectivity, subnetworks)
        by_pair = self._by_pair(projections, subnetworks)
        by_pair[('Subnetwork A', 'Subnetwork A')].scale = 4.0
        by_pair[('Subnetwork A', 'Subnetwork B')].scale = 7.0

        # the whole Connectivity back in the first Subnetwork: the second one is gone, and with it the
        # pair between them
        regrouped = self.service.discard_empty_subnetworks(
            self.service.move_regions(subnetworks, [4, 5, 6, 7], 0))
        reconciled = self._projections(connectivity, regrouped, projections)

        assert len(reconciled) == 1
        assert reconciled[0].is_intra and reconciled[0].scale == 4.0

        # and a Subnetwork added again gets its Projections with the defaults
        again = self.service.move_regions(self.service.add_subnetwork(regrouped), [4, 5, 6, 7], 1)
        reconciled = self._by_pair(self._projections(connectivity, again, reconciled), again)
        assert reconciled[('Subnetwork A', 'Subnetwork A')].scale == 4.0
        assert reconciled[(again[0].name, again[1].name)].scale == 1.0

    def test_a_model_change_resets_the_coupling_variables_it_no_longer_has(self):
        from tvb.simulator.models import JansenRit

        connectivity = self._connectivity()
        subnetworks = self._two_blocks()
        projections = self._projections(connectivity, subnetworks)
        subnetworks[1].dynamics.model = JansenRit()

        reconciled, reset = self.service.reconcile_projections(connectivity, subnetworks, projections)

        by_pair = self._by_pair(reconciled, subnetworks)
        first_jansen_rit = self.service.coupling_variable_names(JansenRit())[0]
        assert by_pair[('Subnetwork B', 'Subnetwork B')].source_cvar == [first_jansen_rit]
        assert by_pair[('Subnetwork A', 'Subnetwork B')].target_cvar == [first_jansen_rit]
        # the Projection staying within the first Subnetwork keeps what it had
        assert by_pair[('Subnetwork A', 'Subnetwork A')].source_cvar == ['V']
        assert {self.service.projection_key(projection) for projection in reset} == \
               {self.service.projection_key(by_pair[('Subnetwork B', 'Subnetwork B')]),
                self.service.projection_key(by_pair[('Subnetwork A', 'Subnetwork B')])}

    def test_a_projection_the_library_can_not_run_is_refused(self):
        connectivity = self._connectivity()
        projection = self._projections(connectivity, self._two_blocks())[0]

        assert self.service.validate_projection(projection) == {}

        projection.source_cvar = ['a', 'b']
        projection.target_cvar = ['c', 'd', 'e']
        projection.cv = 0.0
        errors = self.service.validate_projection(projection)

        # BaseProjection.apply only takes as many on each side, or a single one on either side
        assert set(errors) == {'target_cvar', 'cv'}
        projection.target_cvar = ['c']
        projection.cv = 3.0
        assert self.service.validate_projection(projection) == {}

    def test_excluding_projections_is_reported(self):
        connectivity = self._connectivity()
        subnetworks = self._two_blocks()
        projections = self._projections(connectivity, subnetworks)
        by_pair = self._by_pair(projections, subnetworks)

        # nothing is added yet, so nothing is coupled, which the step says once rather than per Projection
        assert self.service.projection_warnings(subnetworks, projections) == []

        for projection in projections:
            projection.included = True
        assert self.service.projection_warnings(subnetworks, projections) == []

        by_pair[('Subnetwork B', 'Subnetwork B')].included = False
        by_pair[('Subnetwork A', 'Subnetwork B')].included = False
        warnings = self.service.projection_warnings(subnetworks, projections)

        assert len(warnings) == 2
        assert 'Subnetwork B are not coupled to one another' in warnings[0]
        assert 'simulated independently' in warnings[1]
        # the pair the Connectivity holds nothing for has no Projection at all, excluded or not
        assert self.service.pairs_without_projection(subnetworks, projections) == \
               [{'source': 'Subnetwork B', 'target': 'Subnetwork A'}]

    def test_the_projections_are_described_by_name_the_ones_not_added_included(self):
        connectivity = self._connectivity()
        subnetworks = self._two_blocks()
        projections = self._projections(connectivity, subnetworks)
        self._by_pair(projections, subnetworks)[('Subnetwork A', 'Subnetwork A')].included = True

        rows = self.service.describe_projections(connectivity, subnetworks, projections)

        assert [(row['kind'], row['source'], row['target'], row['included']) for row in rows] == [
            ('Intra', 'Subnetwork A', 'Subnetwork A', True),
            ('Intra', 'Subnetwork B', 'Subnetwork B', False),
            ('Inter', 'Subnetwork A', 'Subnetwork B', False)]
        # counted from the Connectivity, so a Projection not added still says what it would carry
        assert [row['connections'] for row in rows] == [16, 16, 16]
        assert rows[2]['coupling'] == 'Linear'
        # 10 mm tracts at 3 mm/ms
        assert abs(rows[2]['max_delay'] - 10.0 / 3.0) < 1e-9
