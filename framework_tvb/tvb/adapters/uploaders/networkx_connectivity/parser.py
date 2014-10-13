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

import numpy
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException
from tvb.datatypes.connectivity import Connectivity


class NetworkxParser():
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


    def __init__(self, storage_path, key_edge_weight=None, key_edge_tract=None, key_node_coordinates=None,
                 key_node_label=None, key_node_region=None, key_node_hemisphere=None):

        self.logger = get_logger(__name__)
        self.storage_path = storage_path

        if key_edge_weight and not key_edge_weight in self.KEY_EDGE_WEIGHT:
            self.KEY_EDGE_WEIGHT.insert(0, key_edge_weight)
        if key_edge_tract and not key_edge_tract in self.KEY_EDGE_TRACT:
            self.KEY_EDGE_TRACT.insert(0, key_edge_tract)
        if key_node_coordinates and not key_node_coordinates in self.KEY_NODE_COORDINATES:
            self.KEY_NODE_COORDINATES.insert(0, key_node_coordinates)
        if key_node_label and not key_node_label in self.KEY_NODE_LABEL:
            self.KEY_NODE_LABEL.insert(0, key_node_label)
        if key_node_region and not key_node_region in self.KEY_NODE_REGION:
            self.KEY_NODE_REGION.insert(0, key_node_region)
        if key_node_hemisphere and not key_node_hemisphere in self.KEY_NODE_HEMISPHERE:
            self.KEY_NODE_HEMISPHERE.insert(0, key_node_hemisphere)


    @staticmethod
    def prepare_input_params_tree(prefix=None):
        """
        :return: Adapter Input tree, with possible user-given keys for reading from networkx object
        """
        result = []
        configurable_keys = {'key_edge_weight': NetworkxParser.KEY_EDGE_WEIGHT,
                             'key_edge_tract': NetworkxParser.KEY_EDGE_TRACT,
                             'key_node_coordinates': NetworkxParser.KEY_NODE_COORDINATES,
                             'key_node_label': NetworkxParser.KEY_NODE_LABEL,
                             'key_node_region': NetworkxParser.KEY_NODE_REGION,
                             'key_node_hemisphere': NetworkxParser.KEY_NODE_HEMISPHERE}

        for init_param in sorted(configurable_keys.keys()):
            label = init_param.replace('_', ' ').capitalize()
            if prefix:
                label = prefix + label
            result.append({'name': init_param, 'label': label, 'default': configurable_keys[init_param][0],
                           'type': 'str', 'required': False})

        return result


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
            for node in network.nodes():
                node_data = network.node[node]

                pos = self._find_value(node_data, self.KEY_NODE_COORDINATES)
                positions.append(list(pos))

                label = self._find_value(node_data, self.KEY_NODE_LABEL)
                labels_vector.append(str(label))

                if self.REGION_CORTICAL == self._find_value(node_data, self.KEY_NODE_REGION):
                    cortical.append(1)
                else:
                    cortical.append(0)

                if self.HEMISPHERE_RIGHT == self._find_value(node_data, self.KEY_NODE_HEMISPHERE):
                    hemisphere.append(True)
                else:
                    hemisphere.append(False)

            # Iterate over edges:
            for start, end in network.edges():
                weights_matrix[start - 1][end - 1] = self._find_value(network.adj[start][end], self.KEY_EDGE_WEIGHT)
                tract_matrix[start - 1][end - 1] = self._find_value(network.adj[start][end], self.KEY_EDGE_TRACT)

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

        msg = "Could not find any of the labels %s" % str(candidate_keys)
        self.logger.error(msg)
        raise ParseException(msg)

