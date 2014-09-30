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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException
from tvb.datatypes.connectivity import Connectivity


class NetworkxParser():
    """
    This class reads content of a NetworkX stream and builds a Connectivity instance filled with details.
    """
    KEY_CMT_COORDINATES = "dn_position"
    KEY_CMT_LABEL = ["dn_name", "dn_label"]

    KEY_CMT_REGION = "dn_region"
    KEY_CMT_REGION_CORTICAL = "cortical"

    KEY_CMT_HEMISPHERE = "dn_hemisphere"
    KEY_CMT_HEMISPHERE_RIGHT = "right"

    KEY_CMT_WEIGHT = "fa_mean"
    KEY_CMT_TRACT = "fiber_length_mean"


    def __init__(self, storage_path):
        self.logger = get_logger(__name__)
        self.storage_path = storage_path


    def parse(self, network):
        """
        Populate Connectivity DataType from NetworkX object.
        Structure inspired from Connectome Mapper Toolkit.

        :param network: NetworkX graph
        :return: Connectivity object
        """
        weights_matrix, tract_matrix, labels_vector = [], [], []
        positions, cortical, hemisphere = [], [], []
        # Read all nodes
        graph_size = len(network.nodes())

        try:
            for node in network.nodes():
                node_data = network.node[node]
                positions.append(list(node_data[self.KEY_CMT_COORDINATES]))

                label = self._find_value(node_data, self.KEY_CMT_LABEL)
                labels_vector.append(str(label))

                weights_matrix.append([0.0] * graph_size)
                tract_matrix.append([0.0] * graph_size)

                if self.KEY_CMT_REGION_CORTICAL == node_data[self.KEY_CMT_REGION]:
                    cortical.append(1)
                else:
                    cortical.append(0)

                if self.KEY_CMT_HEMISPHERE_RIGHT == node_data[self.KEY_CMT_HEMISPHERE]:
                    hemisphere.append(True)
                else:
                    hemisphere.append(False)


            # Read all edges (and make the matrix square
            for edge in network.edges():
                start = edge[0]
                end = edge[1]
                weights_matrix[start - 1][end - 1] = network.adj[start][end][self.KEY_CMT_WEIGHT]
                weights_matrix[end - 1][start - 1] = weights_matrix[start - 1][end - 1]
                tract_matrix[start - 1][end - 1] = network.adj[start][end][self.KEY_CMT_TRACT]
                tract_matrix[end - 1][start - 1] = tract_matrix[start - 1][end - 1]

            result = Connectivity()
            result.storage_path = self.storage_path
            result.region_labels = labels_vector
            result.centres = positions
            result.set_metadata({'description': 'Array Columns: labels, X, Y, Z'}, 'centres')
            result.hemispheres = hemisphere
            result.cortical = cortical
            result.weights = weights_matrix
            result.tract_lengths = tract_matrix
            return result

        except KeyError, err:
            self.logger.exception("Could not parse Connectivity")
            raise ParseException(err)
        

    def _find_value(self, node_data, candidate_keys):
        """
        Find a value in node data using a list of candidate keys
        :return string value or raise ParseException
        """
        for key in candidate_keys:
            if key in node_data:
                return node_data[key]

        msg = "Could not find labels" % str(candidate_keys)
        self.logger.error(msg)
        raise ParseException(msg)

