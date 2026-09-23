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
Service holding the Subnetwork grouping logic used by the Hybrid Simulator UI.

The Subnetworks are kept as :class:`HybridSubnetworkViewModel` instances, storing a stable identifier, a
name, the assigned Connectivity node indices and the configured dynamics (Model and Integrator). The
mapping towards ``tvb.simulator.hybrid.Subnetwork`` is done in a later step of the Hybrid Simulator
workflow.

.. moduleauthor:: TVB Team
"""

import copy
import json
import keyword
import re

import numpy

from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.simulator.view_model import HybridSubnetworkDynamics, HybridSubnetworkViewModel
from tvb.core.neocom import h5
from tvb.core.services.burst_config_serialization import SerializationManager
from tvb.core.services.exceptions import ServicesBaseException
from tvb.simulator.hybrid import NetworkSet
from tvb.simulator.hybrid import Subnetwork as LibrarySubnetwork
from tvb.simulator.hybrid.projection_utils import create_inter_projection, create_intra_projection


class HybridSubnetworkException(ServicesBaseException):
    """
    Exception thrown when the requested Subnetwork operation would leave the Hybrid Simulator
    configuration in an invalid state.
    """


class HybridSimulatorService(object):
    """
    Keeps the Connectivity regions grouped into Subnetworks for the Hybrid Simulator configuration.

    Every operation exposed here preserves the following invariants:
        * there is always at least one Subnetwork;
        * every Connectivity node belongs to exactly one Subnetwork;
        * the node indices are the original Connectivity indices, they are never renumbered;
        * Subnetwork names are non-empty and unique.
    """

    NAME_PREFIX = 'Subnetwork '
    ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)

    # ---------------------------------------------------------------- Connectivity accessors

    @staticmethod
    def get_region_labels(connectivity_gid):
        """
        :return: the list of region labels of the given Connectivity, in the original node order
        """
        with h5.h5_file_for_gid(connectivity_gid) as conn_h5:
            return [str(label) for label in conn_h5.get_region_labels()]

    # ---------------------------------------------------------------- Subnetwork name helpers

    @classmethod
    def _default_name_for(cls, position):
        """
        Build the default name for the Subnetwork found on the given 0-based position: A, B, ... Z, AA, AB, ...
        """
        letters = ''
        position += 1
        while position > 0:
            position, remainder = divmod(position - 1, len(cls.ALPHABET))
            letters = cls.ALPHABET[remainder] + letters
        return cls.NAME_PREFIX + letters

    @classmethod
    def _build_unique_name(cls, subnetworks):
        existing_names = {subnetwork.name for subnetwork in subnetworks}
        position = len(subnetworks)
        while cls._default_name_for(position) in existing_names:
            position += 1
        return cls._default_name_for(position)

    # ---------------------------------------------------------------- Subnetwork operations

    @classmethod
    def create_default_subnetworks(cls, number_of_regions):
        """
        Build the initial configuration: a single Subnetwork holding all the Connectivity nodes.
        """
        return [HybridSubnetworkViewModel(name=cls._default_name_for(0),
                                          node_indices=list(range(number_of_regions)))]

    @classmethod
    def prepare_subnetworks(cls, hybrid_simulator, number_of_regions):
        """
        Return the Subnetworks stored on the given Hybrid Simulator configuration, after making sure they
        still describe a valid partition of the given number of Connectivity nodes. When they do not
        (nothing configured yet, or the Connectivity was changed), the default configuration is created.

        :return: the list of Subnetworks, also stored back on the Hybrid Simulator configuration
        """
        subnetworks = list(hybrid_simulator.subnetworks or [])

        if not cls.is_valid_partition(subnetworks, number_of_regions):
            subnetworks = cls.create_default_subnetworks(number_of_regions)

        hybrid_simulator.subnetworks = subnetworks
        return subnetworks

    @staticmethod
    def is_valid_partition(subnetworks, number_of_regions):
        """
        :return: True only when the given Subnetworks assign every Connectivity node exactly once
        """
        if not subnetworks:
            return False

        assigned = []
        for subnetwork in subnetworks:
            assigned.extend(subnetwork.node_indices)

        return sorted(assigned) == list(range(number_of_regions))

    @staticmethod
    def discard_empty_subnetworks(subnetworks):
        """
        Drop the Subnetworks that ended up holding no region. Empty Subnetworks are useful while
        grouping, as somewhere to drag regions into, but they can not take part in a simulation.
        At least one Subnetwork is always kept, so the configuration stays valid.

        :return: the remaining Subnetworks
        """
        populated = [subnetwork for subnetwork in subnetworks if len(subnetwork.node_indices) > 0]
        return populated or list(subnetworks[:1])

    @staticmethod
    def copy_subnetworks(subnetworks):
        """
        Build an independent copy of the given Subnetworks. The grouping operations change the
        HybridSubnetworkViewModel instances in place, so the draft edited on the board and the saved
        configuration must never share them, or editing the draft would also change what the wizard shows.

        The identifier is carried over: it is what keeps the configured dynamics attached to the intended
        Subnetwork across a copy. The dynamics are deep copied for the same reason the grouping is.
        """
        copies = []
        for subnetwork in subnetworks or []:
            copies.append(HybridSubnetworkViewModel(
                id=subnetwork.id, name=subnetwork.name, node_indices=list(subnetwork.node_indices),
                dynamics=copy.deepcopy(subnetwork.dynamics)))
        return copies

    @classmethod
    def same_grouping(cls, subnetworks, other_subnetworks):
        """
        :return: True when both describe the same Subnetworks, by name and by assigned Connectivity nodes
        """
        return cls.to_json_ready(subnetworks or []) == cls.to_json_ready(other_subnetworks or [])

    @classmethod
    def add_subnetwork(cls, subnetworks):
        """
        Append a new, empty Subnetwork having a generated unique name.
        """
        subnetworks = list(subnetworks)
        subnetworks.append(HybridSubnetworkViewModel(name=cls._build_unique_name(subnetworks), node_indices=[]))
        return subnetworks

    @classmethod
    def rename_subnetwork(cls, subnetworks, subnetwork_index, new_name):
        """
        Change the name of one Subnetwork. Names must be non-empty and unique.
        """
        subnetworks = list(subnetworks)
        cls._check_index(subnetworks, subnetwork_index)

        new_name = (new_name or '').strip()
        if not new_name:
            raise HybridSubnetworkException("The Subnetwork name can not be empty.")

        for index, subnetwork in enumerate(subnetworks):
            if index != subnetwork_index and subnetwork.name == new_name:
                raise HybridSubnetworkException("There is already a Subnetwork named '{}'.".format(new_name))

        subnetworks[subnetwork_index].name = new_name
        return subnetworks

    @classmethod
    def remove_subnetwork(cls, subnetworks, subnetwork_index):
        """
        Remove one Subnetwork. Since no Connectivity node is allowed to remain unassigned, the nodes of the
        removed Subnetwork are moved into the first remaining one. Removing the last Subnetwork is refused.
        """
        subnetworks = list(subnetworks)
        cls._check_index(subnetworks, subnetwork_index)

        if len(subnetworks) == 1:
            raise HybridSubnetworkException("At least one Subnetwork is required, this one can not be removed.")

        removed = subnetworks.pop(subnetwork_index)
        if removed.node_indices:
            fallback = subnetworks[0]
            fallback.node_indices = sorted(list(fallback.node_indices) + list(removed.node_indices))

        return subnetworks

    @classmethod
    def move_regions(cls, subnetworks, node_indices, subnetwork_index):
        """
        Move the given Connectivity nodes into the Subnetwork found on the given position. The nodes keep
        their original Connectivity indices, they are only removed from the Subnetwork currently holding them.
        """
        subnetworks = list(subnetworks)
        cls._check_index(subnetworks, subnetwork_index)

        moved = set()
        for node_index in node_indices or []:
            try:
                moved.add(int(node_index))
            except (TypeError, ValueError):
                raise HybridSubnetworkException("'{}' is not a valid Connectivity node index.".format(node_index))

        if not moved:
            raise HybridSubnetworkException("No Connectivity region was selected to be moved.")

        known = set()
        for subnetwork in subnetworks:
            known.update(subnetwork.node_indices)

        unknown = moved.difference(known)
        if unknown:
            raise HybridSubnetworkException(
                "The Connectivity nodes {} are not part of this Connectivity.".format(sorted(unknown)))

        target = subnetworks[subnetwork_index]
        for index, subnetwork in enumerate(subnetworks):
            if index == subnetwork_index:
                continue
            subnetwork.node_indices = [node for node in subnetwork.node_indices if node not in moved]

        target.node_indices = sorted(set(target.node_indices).union(moved))
        return subnetworks

    # ---------------------------------------------------------------- Subnetwork dynamics

    # Bookkeeping carried by every ViewModel, plus the RandomState of a stochastic Integrator. None of
    # these describes a configuration choice, and gid differs between two copies of the same dynamics,
    # so they are all left out when deciding whether something was actually edited.
    NON_CONFIGURATION_ATTRS = ('operation_group_gid', 'ranges', 'range_values', 'is_metric_operation',
                               'gid', 'create_date', 'random_stream')

    @staticmethod
    def find_subnetwork(subnetworks, subnetwork_id):
        """
        :return: the Subnetwork carrying the given identifier
        :raise HybridSubnetworkException: when no Subnetwork carries it
        """
        for subnetwork in subnetworks or []:
            if subnetwork.id == subnetwork_id:
                return subnetwork
        raise HybridSubnetworkException("This Subnetwork is no longer part of the configuration.")

    @classmethod
    def prepare_dynamics_draft(cls, subnetworks, draft):
        """
        Build the dynamics draft for the given Subnetworks: the dynamics being edited, keyed by Subnetwork
        identifier.

        Entries are seeded from the saved dynamics of every Subnetwork that has none in the draft yet, and
        entries keyed by an identifier that no longer exists are dropped. That is what makes a Subnetwork
        removal, or a regenerated grouping, discard its dynamics instead of reattaching them to whichever
        Subnetwork happens to sit on the same position.

        :return: the draft, a dict of Subnetwork identifier to HybridSubnetworkDynamics
        """
        draft = dict(draft or {})
        known_ids = {subnetwork.id for subnetwork in subnetworks or []}

        for orphan_id in set(draft.keys()).difference(known_ids):
            del draft[orphan_id]

        for subnetwork in subnetworks or []:
            if subnetwork.id not in draft:
                draft[subnetwork.id] = copy.deepcopy(subnetwork.dynamics)

        return draft

    @classmethod
    def apply_shared_dt(cls, dynamics_by_id, dt):
        """
        Give every Integrator the shared step size. tvb.simulator.hybrid.Simulator refuses a NetworkSet
        whose Subnetworks disagree on dt, so it is applied uniformly here rather than being editable per
        Subnetwork.
        """
        for dynamics in (dynamics_by_id or {}).values():
            if dynamics.integrator is not None:
                dynamics.integrator.dt = dt
        return dynamics_by_id

    @classmethod
    def store_dynamics(cls, subnetworks, dynamics_by_id):
        """
        Write the dynamics currently being edited onto the given Subnetworks. This is the only place the
        saved dynamics change, which is what keeps the wizard summary showing the saved configuration
        rather than the one being edited.
        """
        for subnetwork in subnetworks or []:
            dynamics = (dynamics_by_id or {}).get(subnetwork.id)
            if dynamics is not None:
                subnetwork.dynamics = copy.deepcopy(dynamics)
        return subnetworks

    @classmethod
    def same_dynamics(cls, subnetworks, dynamics_by_id):
        """
        :return: True when the dynamics being edited are the ones already saved on the given Subnetworks
        """
        for subnetwork in subnetworks or []:
            dynamics = (dynamics_by_id or {}).get(subnetwork.id)
            if dynamics is None:
                continue
            if cls.dynamics_signature(dynamics) != cls.dynamics_signature(subnetwork.dynamics):
                return False
        return True

    @classmethod
    def dynamics_signature(cls, dynamics):
        """
        :return: a comparable description of one Subnetwork's dynamics, covering the selected Model and
                 Integrator classes and every configured parameter, nested Noise and Equation included
        """
        if dynamics is None:
            return None
        return (cls._trait_signature(dynamics.model), cls._trait_signature(dynamics.integrator))

    @classmethod
    def _trait_signature(cls, trait):
        """
        Describe a HasTraits instance by its class and its declared parameter values, recursing into
        nested traits so that a Noise, or the Equation of a Multiplicative Noise, is covered as well.
        """
        if trait is None:
            return None

        values = []
        for attr_name in sorted(type(trait).declarative_attrs):
            if attr_name in cls.NON_CONFIGURATION_ATTRS:
                continue
            try:
                value = getattr(trait, attr_name)
            except Exception:
                # an unassigned attribute is not a configured value
                continue
            values.append((attr_name, cls._value_signature(value)))

        return type(trait).__name__, tuple(values)

    @classmethod
    def _value_signature(cls, value):
        if isinstance(value, numpy.ndarray):
            return numpy.array2string(value, precision=12, threshold=numpy.inf)
        if hasattr(type(value), 'declarative_attrs'):
            return cls._trait_signature(value)
        if isinstance(value, (list, tuple)):
            return tuple(cls._value_signature(item) for item in value)
        return repr(value)

    @classmethod
    def validate_model_parameters(cls, subnetwork):
        """
        Make sure every Model parameter of the given Subnetwork broadcasts onto the nodes it owns.

        A Model parameter is an array. tvb.simulator.hybrid.Subnetwork applies it to that Subnetwork's own
        nodes, so a single shared value or one value per owned node are the only lengths that can work; an
        array sized for the whole Connectivity, in particular, can not.

        :raise HybridSubnetworkException: naming the Subnetwork, the parameter and the accepted lengths
        """
        model = subnetwork.dynamics.model if subnetwork.dynamics else None
        if model is None:
            return

        nnodes = len(subnetwork.node_indices)
        for attr_name in type(model).declarative_attrs:
            if attr_name in cls.NON_CONFIGURATION_ATTRS:
                continue
            try:
                value = getattr(model, attr_name)
            except Exception:
                continue
            if not isinstance(value, numpy.ndarray) or value.ndim == 0:
                continue
            length = value.shape[0]
            if length not in (1, nnodes):
                raise HybridSubnetworkException(
                    "Subnetwork '{}' owns {} regions, so its Model parameter '{}' needs either 1 value "
                    "shared by all of them or {} values, but {} were given.".format(
                        subnetwork.name, nnodes, attr_name, nnodes, length))

    # ---------------------------------------------------------------- Region Model setup

    @staticmethod
    def dynamics_for_model(dynamics, model):
        """
        The saved Dynamics that can be placed on this Subnetwork's regions: the ones built on the same
        Model class it is configured with.

        Restricting the offer is what keeps the Model class chosen on the wizard step the only place it
        is decided. The classic Cockpit instead overwrites the Simulator's Model with whatever class the
        chosen Dynamics happen to carry.
        """
        model_class_name = type(model).__name__ if model is not None else None
        return [dynamic for dynamic in dynamics or [] if dynamic.model_class == model_class_name]

    @staticmethod
    def parameters_of_dynamic(dynamic):
        """
        :return: the Model parameter values a Dynamic holds, as a plain dict
        """
        return dict(json.loads(dynamic.model_parameters))

    @classmethod
    def apply_dynamics_to_model(cls, model, node_indices, assignment, dynamics_by_id):
        """
        Write the Dynamics placed on this Subnetwork's regions onto its Model, as one array per
        parameter in the Subnetwork's own node order.

        :param node_indices: this Subnetwork's Connectivity node indices, in order
        :param assignment: what each of them is configured with, as node index to Dynamic id
        :param dynamics_by_id: the available Dynamics
        :raise HybridSubnetworkException: when a region has no Dynamic placed on it yet, since a Model
                                          parameter has to hold a value for every node of the Subnetwork
        """
        ordered_parameters = []
        for node_index in node_indices:
            dynamic = dynamics_by_id.get(assignment.get(node_index))
            if dynamic is None:
                raise HybridSubnetworkException(
                    "Every region needs a Model configuration before it can be applied; "
                    "{} of them do not have one yet.".format(cls.unassigned_count(node_indices, assignment)))
            ordered_parameters.append(cls.parameters_of_dynamic(dynamic))

        # the same grouping the classic Set up region Model does, so both produce the same arrays
        grouped = SerializationManager.group_parameter_values_by_name(ordered_parameters)

        for parameter_name, values in grouped.items():
            if not hasattr(type(model), parameter_name):
                # a Dynamic may carry values this Model class does not declare, leave those alone
                continue
            setattr(model, parameter_name, cls._contract_constant(values))

        return model

    @staticmethod
    def _contract_constant(values):
        """
        One value per node, or a single shared one when they are all the same. Both broadcast onto the
        Subnetwork, and the contracted form is what the classic Cockpit stores as well.
        """
        if len(set(values)) == 1:
            values = values[:1]
        return numpy.array(values, dtype=numpy.float64)

    @staticmethod
    def unassigned_count(node_indices, assignment):
        """
        :return: how many of this Subnetwork's regions have no Model configuration placed on them yet
        """
        return len([node_index for node_index in node_indices if assignment.get(node_index) is None])

    @classmethod
    def region_model_rows(cls, node_indices, region_labels, assignment, dynamics_by_id):
        """
        One row per region of this Subnetwork, for the Region Model panel: the original Connectivity
        index, its label, and the Model configuration currently placed on it.
        """
        rows = []
        for node_index in node_indices:
            dynamic = dynamics_by_id.get(assignment.get(node_index))
            rows.append({
                'index': node_index,
                'label': region_labels[node_index] if node_index < len(region_labels) else str(node_index),
                'dynamic_id': dynamic.id if dynamic is not None else None,
                'dynamic_name': dynamic.name if dynamic is not None else ''
            })
        return rows

    @staticmethod
    def place_dynamic_on_regions(assignment, node_indices, dynamic_id):
        """
        Put the given Model configuration on the given regions, leaving the others as they are.

        :return: the updated assignment, as node index to Dynamic id
        """
        assignment = dict(assignment or {})
        for node_index in node_indices or []:
            assignment[node_index] = dynamic_id
        return assignment

    @staticmethod
    def restrict_assignment(assignment, node_indices):
        """
        Drop what was placed on regions this Subnetwork no longer owns, so a regrouping cannot leave a
        stale configuration behind.
        """
        owned = set(node_indices or [])
        return {node_index: dynamic_id for node_index, dynamic_id in (assignment or {}).items()
                if node_index in owned}

    # ---------------------------------------------------------------- tvb_library naming

    @classmethod
    def to_identifiers(cls, subnetworks):
        """
        Map the Subnetwork display names onto valid, unique Python identifiers.

        tvb.simulator.hybrid.NetworkSet builds a namedtuple out of its Subnetworks' names, joining them
        with spaces and splitting the result back into field names, so every name reaching tvb_library has
        to be an identifier. Display names are not (the default 'Subnetwork A' already is not), and
        sanitizing can collide where the display names do not ('Sub A' and 'Sub-A' both yield 'Sub_A'),
        so collisions are disambiguated here.

        :return: a dict of Subnetwork identifier to the tvb_library name to use for it
        """
        identifiers = {}
        used = set()

        for position, subnetwork in enumerate(subnetworks or []):
            candidate = cls._sanitize_identifier(subnetwork.name, position)
            unique = candidate
            suffix = 2
            while unique in used:
                unique = '{}_{}'.format(candidate, suffix)
                suffix += 1
            used.add(unique)
            identifiers[subnetwork.id] = unique

        return identifiers

    @classmethod
    def _sanitize_identifier(cls, name, position):
        candidate = re.sub(r'\W+', '_', (name or '').strip()).strip('_')
        if not candidate or candidate[0].isdigit():
            candidate = 'subnetwork_{}'.format(candidate) if candidate else 'subnetwork_{}'.format(position)
        if keyword.iskeyword(candidate):
            candidate = candidate + '_'
        return candidate

    # ---------------------------------------------------------------- tvb_library translation

    @classmethod
    def build_library_subnetworks(cls, subnetworks, dt):
        """
        Turn the configuration gathered in the web UI into ``tvb.simulator.hybrid.Subnetwork`` objects.

        The Integrator view models subclass the library Integrators, so one can be handed to ``scheme``
        directly. Model and Integrator are deep copied first: the objects on the configuration are the
        ones the forms keep editing, and ``configure`` mutates what it is given.

        :param dt: the shared integration step size, given to every Integrator
        :return: one library Subnetwork per configured Subnetwork, in configuration order
        """
        identifiers = cls.to_identifiers(subnetworks)
        built = []

        for subnetwork in subnetworks or []:
            model = copy.deepcopy(subnetwork.dynamics.model)
            model.configure()

            scheme = copy.deepcopy(subnetwork.dynamics.integrator)
            scheme.dt = dt
            scheme.configure()

            node_indices = numpy.array(sorted(subnetwork.node_indices), dtype=numpy.int_)
            built.append(LibrarySubnetwork(
                # NetworkSet builds a namedtuple out of these, so they have to be identifiers
                name=identifiers[subnetwork.id],
                model=model,
                scheme=scheme,
                nnodes=len(node_indices),
                node_indices=node_indices))

        return built

    @staticmethod
    def default_source_cvar(model):
        """
        :return: the state variable index of the model's first coupling variable, which is what a
                 projection's ``source_cvar`` indexes in the history buffer
        """
        return int(model.cvar[0])

    # A projection's target_cvar is a slot in the target model's cvar list, not a state variable index,
    # so the first coupling variable of any model is always slot 0.
    DEFAULT_TARGET_CVAR = 0

    @classmethod
    def build_network_set(cls, connectivity, subnetworks, dt):
        """
        Build the ``NetworkSet`` the configuration describes: one IntraProjection per Subnetwork and one
        InterProjection per connected ordered pair, all sliced out of the Connectivity.

        Weights and lengths are sliced by ``tvb.simulator.hybrid.projection_utils``, which indexes them
        as ``(target, source)`` - the Connectivity's own orientation - rather than by this service.

        Coupling variables are left at safe defaults: each side uses its own model's first coupling
        variable. Choosing them per projection is follow-up work.

        A pair whose weights block is entirely zero gets no InterProjection: the two Subnetworks are not
        connected in this Connectivity, so a projection would only carry zeros.

        :return: the NetworkSet, configured
        """
        built = cls.build_library_subnetworks(subnetworks, dt)

        for subnet in built:
            subnet.projections = [create_intra_projection(
                subnet,
                source_cvar=cls.default_source_cvar(subnet.model),
                target_cvar=cls.DEFAULT_TARGET_CVAR,
                connectivity=connectivity,
                dt=dt)]
            subnet.configure()

        projections = []
        for source in built:
            for target in built:
                if source is target:
                    continue
                if not cls.are_connected(connectivity, source, target):
                    continue
                projections.append(create_inter_projection(
                    source_subnet=source,
                    target_subnet=target,
                    source_cvar=cls.default_source_cvar(source.model),
                    target_cvar=cls.DEFAULT_TARGET_CVAR,
                    connectivity=connectivity,
                    source_indices=source.node_indices,
                    target_indices=target.node_indices,
                    dt=dt))

        network_set = NetworkSet(subnets=built, projections=projections)
        network_set.configure()
        return network_set

    @staticmethod
    def are_connected(connectivity, source, target):
        """
        :return: True when this Connectivity holds any non zero weight from the source Subnetwork's
                 nodes to the target's
        """
        block = connectivity.weights[numpy.ix_(target.node_indices, source.node_indices)]
        return bool(numpy.any(block))

    @staticmethod
    def describe_network_set(network_set):
        """
        Describe the generated NetworkSet for the wizard step, so the configuration can be inspected
        without leaving the page.
        """
        def cvars(value):
            return [int(index) for index in numpy.atleast_1d(value)]

        rows = []

        for subnet in network_set.subnets:
            for projection in subnet.projections:
                rows.append({
                    'kind': 'Intra',
                    'source': subnet.name,
                    'target': subnet.name,
                    'shape': '{} x {}'.format(*projection.weights.shape),
                    'connections': int(projection.weights.nnz),
                    'source_cvar': cvars(projection.source_cvar),
                    'target_cvar': cvars(projection.target_cvar)
                })

        for projection in network_set.projections:
            rows.append({
                'kind': 'Inter',
                'source': projection.source.name,
                'target': projection.target.name,
                'shape': '{} x {}'.format(*projection.weights.shape),
                'connections': int(projection.weights.nnz),
                'source_cvar': cvars(projection.source_cvar),
                'target_cvar': cvars(projection.target_cvar)
            })

        return rows

    @classmethod
    def unconnected_pairs(cls, network_set):
        """
        :return: the ordered Subnetwork pairs left without an InterProjection, so the step can say so
                 rather than quietly leaving them out
        """
        connected = {(projection.source.name, projection.target.name)
                     for projection in network_set.projections}
        pairs = []
        for source in network_set.subnets:
            for target in network_set.subnets:
                if source is target or (source.name, target.name) in connected:
                    continue
                pairs.append({'source': source.name, 'target': target.name})
        return pairs

    # ---------------------------------------------------------------- Helpers

    @staticmethod
    def _check_index(subnetworks, subnetwork_index):
        if not isinstance(subnetwork_index, int) or subnetwork_index < 0 or subnetwork_index >= len(subnetworks):
            raise HybridSubnetworkException("There is no Subnetwork on position {}.".format(subnetwork_index))

    @staticmethod
    def to_json_ready(subnetworks):
        """
        :return: a JSON serializable representation of the given Subnetworks, as expected by the web UI
        """
        return [{'name': subnetwork.name, 'node_indices': list(subnetwork.node_indices)}
                for subnetwork in subnetworks]
