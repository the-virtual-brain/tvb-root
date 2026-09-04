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
import keyword
import re

import numpy

from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.simulator.view_model import HybridSubnetworkDynamics, HybridSubnetworkViewModel
from tvb.core.neocom import h5
from tvb.core.services.exceptions import ServicesBaseException


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
