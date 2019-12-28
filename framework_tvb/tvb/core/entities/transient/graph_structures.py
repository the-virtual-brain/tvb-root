# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

from tvb.core.entities.storage import dao 

MAX_SHAPE_SIZE = 50
MIN_SHAPE_SIZE = 10

NODE_DATATYPE_TYPE = "datatype"
NODE_OPERATION_TYPE = "operation"
NODE_OPERATION_GROUP_TYPE = "operationGroup"



class NodeData(object):
    """
    Contains the meta-data that will be set on each GRAPH node.
    """

    shape_size = None
    shape_color = None
    shape_type = None
    node_type = None
    node_entity_id = None
    node_subtitle = None


    def __init__(self, shape_size, shape_color, shape_type, node_type, node_entity_id, node_subtitle):
        self.shape_size = shape_size
        self.shape_color = shape_color
        self.shape_type = shape_type
        self.node_type = node_type
        self.node_entity_id = node_entity_id
        self.node_subtitle = node_subtitle


    def to_json(self):
        """
        Returns the JSON-ready representation of this NodeData instance.
        """
        instance_json = {"$dim": self.shape_size,
                         "$color": self.shape_color,
                         "$type": self.shape_type,
                         "node_type": self.node_type,
                         "node_entity_id": self.node_entity_id,
                         "node_subtitle": self.node_subtitle}
        return instance_json


    @staticmethod
    def build_node_for_datatype(datatype_id, node_subtitle, shape_size=MAX_SHAPE_SIZE, is_group=False):

        shape = "circlesGroup" if is_group else "circle"
        return NodeData(shape_size, "#83548B", shape, NODE_DATATYPE_TYPE, datatype_id, node_subtitle)


    @staticmethod
    def build_node_for_operation(operation, group_id=None):

        if group_id:
            entity_id = group_id
            node_type = NODE_OPERATION_GROUP_TYPE
            shape = "squaresGroup"
        else:
            entity_id = operation.id
            node_type = NODE_OPERATION_TYPE
            shape = "square"

        return NodeData(MAX_SHAPE_SIZE, "#660033", shape, node_type, entity_id, str(operation.start_date))



class NodeStructure(object):
    """
    Define the full structure of a graph NODE (including meta-data of type NodeData and node Adjiacences)
    """

    node_gid = None
    name = None
    data = None
    adjacencies = []
    selected = False


    def __init__(self, node_gid, node_name):
        self.node_gid = node_gid
        self.name = node_name


    def to_json(self):
        """
        Returns the JSON-ready representation of this NodeStructure instance.
        """
        instance_json = {"id": self.node_gid,
                         "name": self.name,
                         "data": self.data.to_json() if self.data is not None else {},
                         "adjacencies": [{"nodeFrom": self.node_gid, "nodeTo": adj,
                                          "data": {}} for adj in self.adjacencies]}
        return instance_json


    @staticmethod
    def build_structure_for_datatype(datatype_gid):

        datatype = dao.get_datatype_by_gid(datatype_gid)
        is_group = dao.is_datatype_group(datatype_gid)

        structure = NodeStructure(datatype_gid, datatype.type)
        structure.data = NodeData.build_node_for_datatype(datatype.id, datatype.display_name, is_group=is_group)
        return structure


    @staticmethod
    def build_structure_for_operation(operation):

        algo = dao.get_algorithm_by_id(operation.fk_from_algo)

        structure = NodeStructure(operation.gid, algo.displayname)
        structure.data = NodeData.build_node_for_operation(operation)
        return structure


    @staticmethod
    def build_structure_for_operation_group(operation_group_gid):

        group = dao.get_operationgroup_by_gid(operation_group_gid)
        operation = dao.get_operations_in_group(group.id, only_first_operation=True)
        algo = dao.get_algorithm_by_id(operation.fk_from_algo)

        structure = NodeStructure(operation_group_gid, algo.displayname)
        structure.data = NodeData.build_node_for_operation(operation, group.id)
        return structure


    @staticmethod
    def build_artificial_root_structure(adjacencies_nodes):

        root_structure = NodeStructure("fakeRootNode", "fakeRootNode")
        root_structure.data = NodeData.build_node_for_datatype("fakeRootNode", "Fake root",
                                                               shape_size=MAX_SHAPE_SIZE / 2)
        root_structure.adjacencies = adjacencies_nodes
        return root_structure



class GraphComponent():
    """
    Class used for representing a single component of the graph.
    One GraphComponent holds multiple lists of NodeStructure instances (for multiple layers).

    A GraphComponent will have the following structure::

        input_datatypes             operation_parent
            |                               |
            |                               |
            V                               V
        [operation_parent]          [output_datatypes]             * Currently Selected node
            |                               |
            |                               |
            V                               V
        output_datatypes            in_operations

    """
    input_datatypes = []
    operation_parent = []
    output_datatypes = []
    in_operations = []


    def __init__(self, dt_inputs, parent_op, dt_outputs, op_inputs):
        self.input_datatypes = dt_inputs
        self.operation_parent = parent_op
        self.output_datatypes = dt_outputs
        self.in_operations = op_inputs



class FullGraphStructure():
    """
    This class contains information for the entire graph to be displayed in UI.
    It holds a list of GraphComponent instances (e.g. multiple UPLOAD ops).
    """

    graph_components = []


    def __init__(self, components):
        self.graph_components = components
        self.fill_shape_size()
        self.fill_all_graph_adjiacences()


    def prepare_for_json(self):
        """
        Returns a list of NodeStructure instances to be serialized for browser-client rendering.
        """
        artificial_root_adj = []
        for component in self.graph_components:
            if len(component.input_datatypes):
                artificial_root_adj.extend(self._get_nodes_gids(component.input_datatypes))
            else:
                artificial_root_adj.extend(self._get_nodes_gids(component.operation_parent))

        result_to_serialize = [NodeStructure.build_artificial_root_structure(artificial_root_adj)]
        for component in self.graph_components:
            for level in [component.input_datatypes, component.operation_parent,
                          component.output_datatypes, component.in_operations]:
                for node_structure in level:
                    result_to_serialize.append(node_structure)

        return result_to_serialize


    def fill_all_graph_adjiacences(self):

        for branch in self.graph_components:
            parent_ops_gids = self._get_nodes_gids(branch.operation_parent)
            dt_outputs_gids = self._get_nodes_gids(branch.output_datatypes)
            op_inputs_gids = self._get_nodes_gids(branch.in_operations)

            self._set_nodes_adjacencies(branch.input_datatypes, parent_ops_gids)
            self._set_nodes_adjacencies(branch.operation_parent, dt_outputs_gids)
            self._set_nodes_adjacencies(branch.output_datatypes, op_inputs_gids, True)


    @staticmethod
    def _get_nodes_gids(list_of_nodes):
        return [node.node_gid for node in list_of_nodes]


    @staticmethod
    def _set_nodes_adjacencies(list_of_nodes, adjacencies, only_for_selected_node=False):
        """
        Sets adjacencies for a list of nodes.
        """
        if only_for_selected_node:
            for node in list_of_nodes:
                if node.selected:
                    node.adjacencies = adjacencies
                else:
                    node.adjacencies = []
        else:
            for node in list_of_nodes:
                node.adjacencies = adjacencies


    def fill_shape_size(self):
        """
        Sets the correct size for each node from this graph.
        """
        no_of_dt_inputs = 0
        no_of_parent_op = 0
        no_of_dt_outputs = 0
        no_of_op_inputs = 0

        for branch in self.graph_components:
            no_of_dt_inputs += len(branch.input_datatypes)
            no_of_parent_op += len(branch.operation_parent)
            no_of_dt_outputs += len(branch.output_datatypes)
            no_of_op_inputs += len(branch.in_operations)

        dt_input_size = self._compute_shape_size(no_of_dt_inputs)
        parent_op_size = self._compute_shape_size(no_of_parent_op)
        dt_output_size = self._compute_shape_size(no_of_dt_outputs)
        op_input = self._compute_shape_size(no_of_op_inputs)

        for branch in self.graph_components:
            self._set_nodes_size(branch.input_datatypes, dt_input_size)
            self._set_nodes_size(branch.operation_parent, parent_op_size)
            self._set_nodes_size(branch.output_datatypes, dt_output_size)
            self._set_nodes_size(branch.in_operations, op_input)


    @staticmethod
    def _set_nodes_size(list_of_nodes, shape_size):
        """
        Sets the size for each node from the given list of nodes.
        """
        if not len(list_of_nodes):
            return

        for node in list_of_nodes:
            if node.data is None:
                continue
            node.data.shape_size = shape_size


    @staticmethod
    def _compute_shape_size(no_of_elements):
        """
        no_of_elements - represents the number of nodes that are displayed on a certain level

        We consider that the canvas width is 1000px. With a size of 50 the max number
        of shapes that may be displayed into the canvas are 9 (=> 111px per shape).

        max_shape_size = 50 points <=> 111px
        min_shape_size = 10 points <=> 20px

        1 point <=> 2px => shape_size = (canvas_width) / [(no_of_nodes + 1) * 2]
        """

        if not no_of_elements:
            return MAX_SHAPE_SIZE

        shape_size = 1000 / ((no_of_elements + 1) * 2)
        if shape_size > MAX_SHAPE_SIZE:
            return MAX_SHAPE_SIZE
        elif shape_size < MIN_SHAPE_SIZE:
            return MIN_SHAPE_SIZE
        else:
            return shape_size


