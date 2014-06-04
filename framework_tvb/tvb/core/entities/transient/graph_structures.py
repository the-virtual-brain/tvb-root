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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

from tvb.core.entities.storage import dao 

MAX_SHAPE_SIZE = 50
MIN_SHAPE_SIZE = 10

NODE_OPERATION_GROUP_TYPE = "operationGroup"
NODE_OPERATION_TYPE = "operation"
NODE_DATATYPE_TYPE = "datatype"

DATATYPE_GROUP_SHAPE = "circlesGroup"
DATATYPE_SHAPE = "circle"
DATATYPE_SHAPE_COLOR = "#83548B"
OPERATION_SHAPE = "square"
OPERATION_SHAPE_COLOR = "#660033"
OPERATION_GROUP_SHAPE = "squaresGroup"
OPERATION_GROUP_SHAPE_COLOR = "#660033"

DT_INPUTS_KEY = "dt_inputs"
PARENT_OP_KEY = "parent_op"
DT_OUTPUTS_KEY = "dt_outputs"
OP_INPUTS_KEY = "op_inputs"


class NodeData():
    """
    Contains the data that will be set on each node.
    """
    shape_size = None
    shape_color = None
    shape_type = None
    #aditional data
    dataType = None
    entity_id = None
    subtitle = None

    def __init__(self, shape_size, shape_color, shape_type, dataType, entity_id, subtitle):
        self.shape_size = shape_size
        self.shape_color = shape_color
        self.shape_type = shape_type
        self.dataType = dataType
        self.entity_id = entity_id
        self.subtitle = subtitle


    def to_json(self):
        """
        Returns the JSON representation of this NodeData.
        """
        json = "{"
        json += "  \"$dim\": \"" + str(self.shape_size) + "\""
        json += ", \"$color\": \"" + self.shape_color + "\""
        json += ", \"$type\": \"" + self.shape_type + "\""
        json += ", \"dataType\": \"" + self.dataType + "\""
        json += ", \"entity_id\": \"" + str(self.entity_id) + "\""
        json += ", \"subtitle\": \"" + str(self.subtitle) + "\""
        json += "}"
        return json




class NodeStructure():
    """
    Define the structure of a graph node.

    id - the graph node id. It is set to an entity GID.
    name - the name of the node
    data - it should be an instance of NodeData or None
    adjacencies - a list of graph nodes IDs
    selected - True if this node is the selected one.
    """
    id = None
    name = None
    data = None
    adjacencies = []
    selected = False


    def __init__(self, node_id, node_name):
        self.id = node_id
        self.name = node_name


    def __create_adjacencies_json(self):
        """
        Creates the adjacencies for this node.
        """
        if not len(self.adjacencies):
            return "[]"
        json = "["
        for i, adjacency in enumerate(self.adjacencies):
            if i:
                json += ","
            json += "{"
            json += "  \"nodeFrom\": \"" + str(self.id) + "\""
            json += ", \"nodeTo\": \"" + str(adjacency) + "\""
            json += ", \"data\": {}"
            json += "}"
        json += "]"
        return json


    def to_json(self):
        """
        Returns the JSON representation of this node.
        """
        json = "{"
        json += "  \"id\": \"" + str(self.id) + "\""
        json += ", \"name\":\"" + str(self.name) + "\""

        json += ", \"data\":"
        if self.data is not None:
            json += self.data.to_json()
        else:
            json += "{}"

        json += ", \"adjacencies\": " + self.__create_adjacencies_json()

        json += "}"
        return json



class DatatypeNodeStructure(NodeStructure):
    """
    This class knows how to create a NodeStructure for a given DataType.
    """
    def __init__(self, datatype_gid):
        NodeStructure.__init__(self, datatype_gid, "")

        datatype_shape = DATATYPE_SHAPE
        if dao.is_datatype_group(datatype_gid):
            datatype_shape = DATATYPE_GROUP_SHAPE
        datatype = dao.get_datatype_by_gid(datatype_gid)

        node_data = NodeData(MAX_SHAPE_SIZE, DATATYPE_SHAPE_COLOR, datatype_shape,
                             NODE_DATATYPE_TYPE, datatype.id, datatype.display_name)

        self.name = str(datatype.type)
        self.data = node_data



class OperationNodeStructure(NodeStructure):
    """
    This class knows how to create a NodeStructure for a given Operation.
    """
    def __init__(self, operation_gid):
        NodeStructure.__init__(self, operation_gid, "")

        operation = dao.get_operation_by_gid(operation_gid)
        algo = dao.get_algorithm_by_id(operation.fk_from_algo)
        node_data = NodeData(MAX_SHAPE_SIZE, OPERATION_SHAPE_COLOR, OPERATION_SHAPE,
                             NODE_OPERATION_TYPE, operation.id, str(operation.start_date))

        self.name = algo.name
        self.data = node_data



class OperationGroupNodeStructure(NodeStructure):
    """
    This class knows how to create a NodeStructure for a given OperationGroup.
    """
    def __init__(self, operation_group_gid):
        NodeStructure.__init__(self, operation_group_gid, "")

        group = dao.get_operationgroup_by_gid(operation_group_gid)
        operation = dao.get_operations_in_group(group.id, only_first_operation=True)
        algo = dao.get_algorithm_by_id(operation.fk_from_algo)
        node_data = NodeData(MAX_SHAPE_SIZE, OPERATION_GROUP_SHAPE_COLOR, OPERATION_GROUP_SHAPE,
                             NODE_OPERATION_GROUP_TYPE, group.id, str(operation.start_date))

        self.name = algo.name
        self.data = node_data



class GraphBranch():
    """
    Class used for representing a branch of the graph.
    If we have executed more than one upload operation than the graph
    will have more branches, one for each of those operations.

    dt_inputs, parent_op, dt_outputs, op_inputs - should be lists of NodeStructure instances.

    A branch will have the following structure::

                            dt_input ... dt_input
                                \\         /
                                 \\       /
                                 parent_op
                                  /       \\
                                /          \\
                            dt_output ... dt_output(if selected)
                                               /         \\
                                             /            \\
                                          op_input ... op_input
    
    """
    dt_inputs = []
    parent_op = []
    dt_outputs = []
    op_inputs = []


    def __init__(self, dt_inputs, parent_op, dt_outputs, op_inputs):
        self.dt_inputs = dt_inputs
        self.parent_op = parent_op
        self.dt_outputs = dt_outputs
        self.op_inputs = op_inputs


    def get_branch_levels(self):
        return {DT_INPUTS_KEY: self.dt_inputs, PARENT_OP_KEY: self.parent_op,
                DT_OUTPUTS_KEY: self.dt_outputs, OP_INPUTS_KEY: self.op_inputs}




class GraphStructure():
    """
    This class contains information for the entire graph.

    graph_branches - represents a list of GraphBranch
    """

    graph_branches = []

    def __init__(self, graph_branches):
        self.graph_branches = graph_branches
        self._set_shapes_size()
        self._set_adjacencies()


    def to_json(self):
        """
        Returns the json representation of the graph.
        """
        fake_root_adjacencies = []
        for branch in self.graph_branches:
            branch_levels = branch.get_branch_levels()
            if len(branch_levels[DT_INPUTS_KEY]):
                fake_root_adjacencies.extend(self._get_nodes_ids(branch_levels[DT_INPUTS_KEY]))
            else:
                fake_root_adjacencies.extend(self._get_nodes_ids(branch_levels[PARENT_OP_KEY]))

        final_json = "["
        final_json += self._create_fake_root(fake_root_adjacencies).to_json()
        for branch in self.graph_branches:
            branch_levels = branch.get_branch_levels()

            for node in branch_levels[DT_INPUTS_KEY]:
                final_json += "," + node.to_json()
            for node in branch_levels[PARENT_OP_KEY]:
                final_json += "," + node.to_json()
            for node in branch_levels[DT_OUTPUTS_KEY]:
                final_json += "," + node.to_json()
            for node in branch_levels[OP_INPUTS_KEY]:
                final_json += "," + node.to_json()

        final_json += "]"
        return final_json


    def _set_adjacencies(self):
        """
        Sets adjacencies for the entire graph.
        """
        for branch in self.graph_branches:
            branch_levels = branch.get_branch_levels()

            parent_ops_ids = self._get_nodes_ids(branch_levels[PARENT_OP_KEY])
            dt_outputs_ids = self._get_nodes_ids(branch_levels[DT_OUTPUTS_KEY])
            op_inputs_ids = self._get_nodes_ids(branch_levels[OP_INPUTS_KEY])

            self.__set_nodes_adjacencies(branch_levels[DT_INPUTS_KEY], parent_ops_ids)
            self.__set_nodes_adjacencies(branch_levels[PARENT_OP_KEY], dt_outputs_ids)
            self.__set_nodes_adjacencies(branch_levels[DT_OUTPUTS_KEY], op_inputs_ids, True)


    @staticmethod
    def __set_nodes_adjacencies(list_of_nodes, adjacencies, only_for_selected_node=False):
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


    def _set_shapes_size(self):
        """
        Sets the correct size for each node from this graph.
        """
        no_of_dt_inputs = 0
        no_of_parent_op = 0
        no_of_dt_outputs = 0
        no_of_op_inputs = 0
        for branch in self.graph_branches:
            branch_levels = branch.get_branch_levels()

            no_of_dt_inputs += len(branch_levels[DT_INPUTS_KEY])
            no_of_parent_op += len(branch_levels[PARENT_OP_KEY])
            no_of_dt_outputs += len(branch_levels[DT_OUTPUTS_KEY])
            no_of_op_inputs += len(branch_levels[OP_INPUTS_KEY])

        dt_input_size = self._compute_shape_size(no_of_dt_inputs)
        parent_op_size = self._compute_shape_size(no_of_parent_op)
        dt_output_size = self._compute_shape_size(no_of_dt_outputs)
        op_input = self._compute_shape_size(no_of_op_inputs)

        for branch in self.graph_branches:
            branch_levels = branch.get_branch_levels()

            self.__set_nodes_size(branch_levels[DT_INPUTS_KEY], dt_input_size)
            self.__set_nodes_size(branch_levels[PARENT_OP_KEY], parent_op_size)
            self.__set_nodes_size(branch_levels[DT_OUTPUTS_KEY], dt_output_size)
            self.__set_nodes_size(branch_levels[OP_INPUTS_KEY], op_input)


    @staticmethod
    def __set_nodes_size(list_of_nodes, shape_size):
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
    def _get_nodes_ids(list_of_nodes):
        """Compute IDs for nodes"""
        nodes_ids = []
        for node in list_of_nodes:
            nodes_ids.append(node.id)
        return nodes_ids


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


    @staticmethod
    def _create_fake_root(adjacencies_nodes):
        """In case no root exists, create one, for JS limitations"""
        node_data = NodeData(7, DATATYPE_SHAPE_COLOR, DATATYPE_SHAPE, 
                             NODE_DATATYPE_TYPE, "fakeRootNode", "Fake root")
        fake_root = NodeStructure("fakeRootNode", "fakeRootNode")
        fake_root.data = node_data
        fake_root.adjacencies = adjacencies_nodes
        return fake_root



