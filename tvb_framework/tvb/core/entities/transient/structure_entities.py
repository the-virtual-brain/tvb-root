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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json
from tvb.basic.config.utils import EnhancedDictionary
from tvb.basic.profile import TvbProfile


class StructureNode:
    """
    This entity represents a node in the Tree of a Project related Structure.
    """
    TYPE_FOLDER = "Folder"
    TYPE_FILE = "File"
    TYPE_INVALID = "Invalid"

    PREFIX_ID_NODE = "node_"
    PREFIX_ID_LEAF = "leaf_"
    PREFIX_ID_PROJECT = "projectID"

    SEP = "__"

    def __init__(self, nid, node_name, ntype=TYPE_FOLDER, meta=None, children=None):
        """
        Constructor 
        
        :param nid: Node Identifier (needs to be UQ)
        :param node_name: Node Display Name 
        :param ntype: Node Type (influences the display mode in UI).
        :param meta: DataTypeMetaData instance.
        :param children: List of StructureNode entities or None.
          
        """
        self.id = nid
        self.name = node_name
        self._type = ntype
        self.metadata = meta
        self.children = children

    @property
    def has_children(self):
        """
        Return TRUE when current node is with Child nodes.
        Return FALSE when current node is a leaf in Tree.
        """
        return self.children and len(self.children) > 0

    @property
    def is_link(self):
        """
        Check if meta_data has the transient is_link attribute set.
        """
        return (self.metadata is not None
                and DataTypeMetaData.KEY_LINK in self.metadata and
                self.metadata[DataTypeMetaData.KEY_LINK] > 0)

    @property
    def is_irelevant(self):
        """Check that current node is marked as not-relevant."""
        if (self.metadata is not None and DataTypeMetaData.KEY_RELEVANCY in self.metadata and
                not self.metadata[DataTypeMetaData.KEY_RELEVANCY]):
            return True
        return False

    @property
    def is_group(self):
        """
        Check if meta_data has the transient is_link attribute set.
        """
        return (self.metadata is not None
                and DataTypeMetaData.KEY_OP_GROUP_ID in self.metadata and
                self.metadata[DataTypeMetaData.KEY_OP_GROUP_ID] is not None)

    @property
    def type(self):
        """
        Type of a node it can be FOLDER, FILE or INVALID.
        """
        return self._type

    @staticmethod
    def metadata2tree(metadatas, first_level, second_level, project_id, project_name):
        """ 
        Take a list of DataTypeMetaData entities as input.
        Create a tree of StructureNode entities, then convert it in JSON to display in UI.

        first_level and second_level represents the fields by which
        should be structured the tree. Those fields should exists
        into the data dict of each DataTypeMetaData object.
        """
        levels = {}
        for meta in metadatas:
            level_one = str(meta[first_level])
            level_two = str(meta[second_level])
            if level_one not in levels:
                levels[level_one] = {level_two: [meta]}
            else:
                existent_sublevels = levels[level_one]
                if level_two not in existent_sublevels:
                    levels[level_one][level_two] = [meta]
                else:
                    levels[level_one][level_two].append(meta)

        forest = []
        for level in sorted(levels):
            sublevels = levels[level]
            level_children = []
            for sublevel in sorted(sublevels):
                metas = sublevels[sublevel]
                datas = []
                for meta in metas:
                    parent_name = meta[meta.KEY_TITLE] + " "

                    if meta[meta.KEY_OPERATION_TAG]:
                        parent_name = parent_name + " - " + meta[meta.KEY_OPERATION_TAG]

                    parent = StructureNode(meta.gid, parent_name, meta=meta)
                    if meta.invalid:
                        parent._type = StructureNode.TYPE_INVALID
                    else:
                        parent._type = meta[meta.KEY_NODE_TYPE]
                    datas.append(parent)
                sublevel_id = level.replace(" ", "") + StructureNode.SEP + sublevel.replace(" ", "")
                sublevel_name = StructureNode._prepare_node_name(sublevel, second_level)
                sublevel_node = StructureNode(sublevel_id, sublevel_name, children=datas)
                sublevel_node._type = StructureNode._capitalize_first_letter(second_level)
                level_children.append(sublevel_node)

            dir_name = StructureNode._prepare_node_name(level, first_level)
            level_node = StructureNode(level, dir_name, children=level_children)
            level_node._type = StructureNode._capitalize_first_letter(first_level)
            forest.append(level_node)

        json_children = StructureNode.__convert2json(forest, project_id)
        if len(json_children) > 0:
            result = '{data: [{ data: {title: "' + project_name + '"'
            result += f',icon: "{TvbProfile.current.web.DEPLOY_CONTEXT}/static/style/nodes/nodeRoot.png"}},'
            result += 'state:"open", attr:{id:"' + StructureNode.PREFIX_ID_PROJECT
            result += '"}, children: [' + json_children + '] } ] }'
        else:
            result = '{data: [{ data: {title: "' + project_name + '"'
            result += f',icon: "{TvbProfile.current.web.DEPLOY_CONTEXT}/static/style/nodes/nodeRoot.png"}}'
            result += ',attr:{id:"' + StructureNode.PREFIX_ID_PROJECT + '"}}]}'

        return result

    @staticmethod
    def _prepare_node_name(name, level_filter):
        """
        Process name. In case filter corresponds to STATE, then return display value for it.
        """
        if level_filter == DataTypeMetaData.KEY_STATE:
            return DataTypeMetaData.STATES[name]
        return name

    @staticmethod
    def _capitalize_first_letter(word):
        """
        :returns: same string, but with first letter capitalized.
        """
        return word[0].upper() + word[1:]

    @staticmethod
    def __convert2json(nodes_list, project_id):
        """
        Local method, for converting an internal Tree structure (of StructureNode entities)
        into a JSON object, ready for display into UI with JSTree.
        """
        result = ""
        place_comma = False
        for node in nodes_list:
            json_node = '{data: {title:"' + (node.name if len(node.name) < 100 else node.name[:95] + "...")
            json_node += '",icon: "{}/static/style/nodes/node'.format(TvbProfile.current.web.DEPLOY_CONTEXT)
            if node.is_group:
                json_node += 'Group.png"},'
            else:
                json_node += node.type + '.png"},'
            if node.metadata is None and node.has_children:
                json_node += ' state:"open", '

            json_node += 'attr:{id:"' + StructureNode.PREFIX_ID_NODE + node.id + '", separator: ">>", '
            json_node += 'projectId:"' + str(project_id) + '"'
            if node.metadata is not None:
                meta_str = json.dumps(node.metadata)
                meta_str = meta_str.replace('{', '').replace('}', '')
                json_node += ',' + meta_str

            json_node += ', style: "'
            if node.is_irelevant:
                json_node += 'background-color: #666;'
            if node.is_link:
                json_node += 'font-style: italic;'
            json_node += '" }'

            if node.has_children:
                json_node += ', children:[' + StructureNode.__convert2json(node.children, project_id) + ']'

            json_node += '}'

            if place_comma:
                result += ','
            place_comma = True

            result += json_node
        return result


class DataTypeMetaData(dict):
    """
    This object will be populated from meta-data stored on a particular DataType/Operation.
    It should contain enough information, to restore a DataType entity, without DB previous data required.
    """
    KEY_GID = "Gid"
    KEY_STATE = "Data_State"
    STATES = {'RAW_DATA': 'Raw Data',
              'INTERMEDIATE': 'Intermediate',
              'FINAL': 'Final'}
    KEY_SUBJECT = "Data_Subject"
    DEFAULT_SUBJECT = "John Doe"
    KEY_BURST = "Burst_Reference"
    KEY_TAG_1 = "User_Tag_1_Perpetuated"
    KEY_TAG_2 = "User_Tag_2"
    KEY_TAG_3 = "User_Tag_3"
    KEY_TAG_4 = "User_Tag_4"
    KEY_TAG_5 = "User_Tag_5"
    KEY_MODULE = "Module"
    KEY_CLASS_NAME = "Type"
    KEY_AUTHOR = "Source_Author"
    KEY_OPERATION_TYPE = "Source_Operation_Category"
    KEY_DATE = "create_date"
    KEY_OPERATION_TAG = "user_group"
    KEY_OP_GROUP_ID = "groupId"
    KEY_RELEVANCY = "Relevant"
    KEY_TITLE = "title"

    # Transient attributes
    KEY_NODE_TYPE = "DataType"
    KEY_DATATYPE_ID = "Datatypeid"
    KEY_INVALID = "datatype_invalid"
    KEY_COUNT = "count"
    KEY_LINK = "link"
    KEY_CREATE_DATA_MONTH = "createDataMonth"
    KEY_CREATE_DATA_DAY = "createDataDay"

    # operation details
    KEY_OPERATION_GROUP_NAME = "op_group_name"
    KEY_OPERATION_ALGORITHM = "Operation_Algorithm"
    KEY_FK_OPERATION_GROUP = 'fk_operation_group'

    def __init__(self, data=None, invalid=False):
        self.invalid = invalid

        if data is not None:
            self.update(data)

    @property
    def gid(self):
        """
        :returns: current Global Identifier or None.
        """
        if self.KEY_GID in list(self):
            return self[self.KEY_GID]
        return None

    @property
    def subject(self):
        """
        Return the name of the Subject if defined, or Default "John Doe".
        """
        if self.KEY_SUBJECT not in list(self):
            return self.DEFAULT_SUBJECT
        return self[self.KEY_SUBJECT]

    @property
    def create_date(self):
        """
        Return the create date of this entity, if stored, or None otherwise.
        """
        if self.KEY_DATE in list(self):
            return self[self.KEY_DATE]
        return None

    @property
    def group(self):
        """
        Return the group tag in which the operation was launched
        """
        if self.KEY_OPERATION_TAG in list(self):
            return self[self.KEY_OPERATION_TAG]
        return None

    def mark_invalid(self):
        """
        Mark current meta-data as invalid.
        e.g. Because of a missing associated file.
        """
        self.invalid = True

    def merge_data(self, new_data):
        """
        Update current state, from an external dictionary.
        """
        if new_data is None:
            return
        for val in new_data:
            # Ignore transient entities
            if val != self.KEY_COUNT and val != self.KEY_OP_GROUP_ID:
                self[val] = new_data[val]

    @classmethod
    def get_filterable_meta(cls):
        """
        Contains all the attributes by which the user can structure the tree of DataTypes.

        All the returned attributes should exists into the 'data' field of its
        corresponding DataTypeMetaData object.
        """
        return [
            {cls.KEY_NODE_TYPE: "Data Type"},
            {cls.KEY_SUBJECT: "Subject"},
            {cls.KEY_STATE: "State"},
            {cls.KEY_OPERATION_TAG: "Operation group name"},
            {cls.KEY_GID: "Unique Global Identifier"},
            {cls.KEY_DATATYPE_ID: "Entity id"},
            {cls.KEY_CREATE_DATA_DAY: "Create data day"},
            {cls.KEY_CREATE_DATA_MONTH: "Create data month"},
            {cls.KEY_OPERATION_ALGORITHM: "Operation algorithm"},
            {cls.KEY_BURST: "Simulation name"},
            {cls.KEY_TAG_1: "DataType Tag 1"},
            {cls.KEY_TAG_2: "DataType Tag 2"},
            {cls.KEY_TAG_3: "DataType Tag 3"},
            {cls.KEY_TAG_4: "DataType Tag 4"},
            {cls.KEY_TAG_5: "DataType Tag 5"}
        ]
