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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import re
import numpy
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException
from tvb.datatypes.connectivity import Connectivity


class NetworkxParser(object):
    """
    This class reads content of a NetworkX stream and builds a Connectivity instance filled with details.
    """

    KEY_NODE_COORDINATES = ["dn_position"]
    KEY_NODE_LABEL = ["dn_name", "dn_label"]

    KEY_NODE_REGION = ["dn_region"]
    REGION_CORTICAL = "cortical"

    KEY_NODE_HEMISPHERE = ["dn_hemisphere"]
    HEMISPHERE_RIGHT = "right"

    KEY_EDGE_WEIGHT = ["adc_mean", "fiber_weight_mean"]
    KEY_EDGE_TRACT = ["fiber_length_mean"]

    OPERATORS = "[*-+:]"

    def __init__(self, view_model):

        self.logger = get_logger(__name__)

        NetworkxParser._append_key(view_model.key_edge_weight, self.KEY_EDGE_WEIGHT)
        NetworkxParser._append_key(view_model.key_edge_tract, self.KEY_EDGE_TRACT)
        NetworkxParser._append_key(view_model.key_node_coordinates, self.KEY_NODE_COORDINATES)
        NetworkxParser._append_key(view_model.key_node_label, self.KEY_NODE_LABEL)
        NetworkxParser._append_key(view_model.key_node_region, self.KEY_NODE_REGION)
        NetworkxParser._append_key(view_model.key_node_hemisphere, self.KEY_NODE_HEMISPHERE)

    def parse(self, network):
        """
        Populate Connectivity DataType from NetworkX object.
        Tested with results from Connectome Mapper Toolkit.

        :param network: NetworkX graph
        :return: Connectivity object
        """
        graph_size = len(network.nodes())

        weights_matrix = numpy.zeros((graph_size, graph_size))
        tract_matrix = numpy.zeros((graph_size, graph_size))
        labels_vector, positions, cortical, hemisphere = [], [], [], []

        try:
            for node in range(1, graph_size + 1):
                node_data = network.nodes[node]

                pos = self._find_value(node_data, self.KEY_NODE_COORDINATES)
                positions.append(list(pos))

                label = self._find_value(node_data, self.KEY_NODE_LABEL)
                labels_vector.append(str(label))

                if self.REGION_CORTICAL == self._find_value(node_data, self.KEY_NODE_REGION):
                    cortical.append(True)
                else:
                    cortical.append(False)

                if self.HEMISPHERE_RIGHT == self._find_value(node_data, self.KEY_NODE_HEMISPHERE):
                    hemisphere.append(True)
                else:
                    hemisphere.append(False)

            # Iterate over edges:
            for start, end in network.edges():
                weights_matrix[start - 1][end - 1] = self._find_value(network.adj[start][end], self.KEY_EDGE_WEIGHT)
                tract_matrix[start - 1][end - 1] = self._find_value(network.adj[start][end], self.KEY_EDGE_TRACT)

            result = Connectivity()
            result.region_labels = numpy.array(labels_vector)
            result.centres = numpy.array(positions)
            # result.set_metadata({'description': 'Array Columns: labels, X, Y, Z'}, 'centres')
            result.hemispheres = numpy.array(hemisphere)
            result.cortical = numpy.array(cortical)
            result.weights = weights_matrix
            result.tract_lengths = tract_matrix
            result.configure()
            return result

        except KeyError as err:
            self.logger.exception("Could not parse Connectivity")
            raise ParseException(err)

    @staticmethod
    def _append_key(key, current_keys):
        """
        This will append to the CLASS attribute the "key".
        It will help us, by storing it for future usages of the same importer (like  a cache),
        as it is highly probable to have the same CFF types on the same TVB installation.
        It is not a strong requirement, so it is fine to be lost at the next TVB restart.
        """
        if key and key not in current_keys:
            current_keys.insert(0, key)

    def _find_value(self, node_data, candidate_keys):
        """
        Find a value in node data using a list of candidate keys
        :return string value or raise ParseException
        """
        for key in candidate_keys:
            if key in node_data:
                return node_data[key]

            else:
                # Try to parse a simple number
                try:
                    return float(key)
                except ValueError:
                    pass

                # Try to parse a chain of operations between multiple keys
                try:
                    split_keys = re.split(self.OPERATORS, key)
                    operators = re.findall(self.OPERATORS, key)
                    if len(split_keys) < 2 or len(split_keys) != len(operators) + 1:
                        continue

                    value = self._find_value(node_data, [split_keys[0]])
                    for i in range(0, len(operators)):
                        expression = "value" + operators[i] + str(self._find_value(node_data, [split_keys[i + 1]]))
                        value = eval(expression)
                    return value
                except ParseException:
                    continue

        msg = "Could not find any of the labels %s in value %s" % (str(candidate_keys), str(node_data))
        self.logger.error(msg)
        raise ParseException(msg)
