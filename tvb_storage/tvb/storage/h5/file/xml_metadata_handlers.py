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
This module contains logic for meta-data handling.

It handles read/write operations in XML files for retrieving/storing meta-data.
More specific: it contains XML Reader/Writer Utility, for generic metadata dictionary.

.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import json
import xml.dom.minidom
from xml.dom.minidom import Node, Document

from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger


class XMLReader(object):
    """
    Reader for XML with meta-data on generic entities (e.g. Project, Operation).
    """

    def __init__(self, xml_path):
        self.logger = get_logger(self.__class__.__module__)
        self.xml_path = xml_path

    def read_metadata_from_xml(self):
        """
        Return a dictionary, filled with data read from XML file.
        """
        self.logger.debug("Starting to parse XML file " + self.xml_path)
        root_node = self._find_root()
        # Parse all nodes, and read text content.    
        result_data = self._parse_xml_node_to_dict(root_node)
        return result_data

    @staticmethod       
    def get_node_text(node):
        """
        From XMl node, read string content.
        """
        for text_child in node.childNodes:
            if text_child.nodeType == Node.TEXT_NODE:
                return str(text_child.data).lstrip().rstrip()
        
        return ''

    ####### PRIVATE METHODS Start Here #######################################

    def _find_root(self):
        """
        From given file path, get XML root node.
        """
        doc_xml = xml.dom.minidom.parse(self.xml_path)
        for child_node in doc_xml.childNodes:
            if child_node.nodeType == Node.ELEMENT_NODE:
                return child_node
        return None

    def _parse_xml_node_to_dict(self, root_node):
        """
        Parse a given input XML node, and return the dictionary of text attributes.
        The dictionary can have multiple levels of deepness, but when child-nodes are 
        encountered, text value of the node is ignored.
        """
        result = {}
        for node in root_node.childNodes:
            if node.nodeType == Node.ELEMENT_NODE:
                result[node.nodeName] = self.get_node_text(node)
                result_meta = self._parse_xml_node_to_dict(node)
                if list(result_meta) is not None and len(list(result_meta)) > 0:
                    result[node.nodeName] = result_meta
        return result
    

class XMLWriter(object):
    """
    Writer for XML with meta-data on generic entities (e.g. Project, Operation).
    """
    ELEM_ROOT = "tvb_data"
    FILE_EXTENSION = ".xml"

    def __init__(self, entity):
        """
        :param entity:  a dictionary that contains generic metadata to be written
        """
        self.entity = entity
        self.logger = get_logger(self.__class__.__module__) 

    def write_metadata_in_xml(self, final_path):
        """
        From a meta-data dictionary for an entity, create the XML file.
        """
        doc = Document()
        root_node = doc.createElement(self.ELEM_ROOT)
        
        # Add information about version of the stored data
        root_node.setAttribute(TvbProfile.current.version.DATA_VERSION_ATTRIBUTE,
                               str(TvbProfile.current.version.DATA_VERSION))
        
        # Add each generic metadata attribute as XML node-elements.
        for att_name, att_value in self.entity.items():
            node = doc.createElement(att_name)
            if isinstance(att_value, list):
                att_value = json.dumps(att_value)
            node.appendChild(doc.createTextNode(str(att_value)))
            root_node.appendChild(node)
        doc.appendChild(root_node)

        # Now dump the XML content into a file.
        with open(final_path, 'wt') as file_obj:
            doc.writexml(file_obj, addindent="\t", newl="\n")
