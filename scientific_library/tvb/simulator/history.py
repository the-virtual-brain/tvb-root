# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Simulator history implementation.

For now it contains only functions to fetch from a history buffer.
In the future it makes sense to have classes that encapsulate the history buffer and querying strategies.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

try:
    import tvb._speedups.history as chist
    LOG.info('Using C speedups for history')

    def get_state(history, time_idx, cvar, node_ids, out):
        """
        Fetches a delayed state from history
        :param history: History array. (time, state_vars, nodes, modes)
        :param time_idx: Delay indices. (nodes, 1 nodes)
        :param cvar: Coupled vars indices. (1, ncvar, 1)
        :param: out: The delayed states (nodes, ncvar, nodes, modes)
        """
        chist.get_state(history, time_idx, cvar, out)

    def _get_state_mask(history, time_idx, cvar, conn_mask, out):
        """
        Fetches a delayed state from history. Uses a mask to avoid fetching history for uncoupled nodes. Faster than get_state
        :param history: History array. (time, state_vars, nodes, modes)
        :param time_idx: Delay indices. (nodes, 1 nodes)
        :param cvar: Coupled vars indices. (1, ncvar, 1)
        :param conn_mask: Should be 0 where the weights are 0 1 otherwise.(nodes, nodes)
        :param: out: The delayed states (nodes, ncvar, nodes, modes)
        """
        chist.get_state_with_mask(history, time_idx, cvar, conn_mask, out)

except ImportError:
    LOG.info('Using the python reference implementation for history')

    def get_state(history, time_idx, cvar, node_ids, out):
        """
        Fetches a delayed state from history
        :param history: History array. (time, state_vars, nodes, modes)
        :param time_idx: Delay indices. (nodes, 1 nodes)
        :param cvar: Coupled vars indices. (1, ncvar, 1)
        :param: out: The delayed states (nodes, ncvar, nodes, modes)
        """
        out[...] = history[time_idx, cvar, node_ids, :]
