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

The Subnetworks are kept as :class:`HybridSubnetworkViewModel` instances, which only store a name and
the assigned Connectivity node indices. The mapping towards ``tvb.simulator.hybrid.Subnetwork`` is done
in a later step of the Hybrid Simulator workflow.

.. moduleauthor:: TVB Team
"""

from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.simulator.view_model import HybridSubnetworkViewModel
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
