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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import xml.dom.minidom
from xml.dom.minidom import Node
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.constants import *

KEY_DYNAMIC = 'dynamic'
KEY_STATIC = 'static'

ATT_MODULE = "module"
ATT_CHAIN = 'chain'
ATT_OVERWRITE = 'overwrite'


class XMLPortletReader(object):
    """
    Helper class to read from XML, a group of Portlets definition.
    """
    logger = get_logger(__name__)

    def __init__(self, interface_file):

        self.algorithms = dict()
        self.logger.info("Starting to parse XML file: " + interface_file)
        doc_xml = xml.dom.minidom.parse(interface_file)
        tvb_node = doc_xml.lastChild

        for child in tvb_node.childNodes:
            if child.nodeName == ELEM_PORTLET:
                alg = self._parse_algorithm(child)
                if alg is not None:
                    self.algorithms[alg.identifier] = alg


    def get_algorithms_dictionary(self):
        """ Retrieve the dictionary of available sub-Algorithm references."""
        result = dict()
        for key, value in self.algorithms.items():
            result[key] = value.to_dict()
        return result


    def get_adapters_chain(self, algorithm_identifier):
        """ Return the list of outputs for a given algorithm"""
        alg = self.algorithms.get(algorithm_identifier, None)
        if alg is None:
            return []
        return alg.chain_adapters


    def get_inputs(self, algorithm_identifier):
        """Return the interface Tree for a single sub-algorithm"""
        alg = self.algorithms.get(algorithm_identifier, None)
        if alg is None:
            return []
        return alg.inputs


    def _parse_algorithm(self, node):
        """"Method used for parsing an algorithm node"""
        algorithm = PortletWrapper()
        algorithm.name = node.getAttribute(ATT_NAME)
        algorithm.identifier = node.getAttribute(ATT_IDENTIFIER)
        if len(node.childNodes) > 0:
            for child in node.childNodes:
                if child.nodeName == ELEM_INPUTS:
                    all_inputs = self._parse_inputs(child.childNodes)
                    chain_adapters = []
                    dynamic_inputs = []
                    default_inputs = []
                    for one_input in all_inputs:
                        if one_input[ATT_NAME].startswith(ATT_CHAIN):
                            chain_adapters.append(one_input)
                        elif one_input[ATT_OVERWRITE] in one_input:
                            dynamic_inputs.append(one_input)
                        else:
                            default_inputs.append(one_input)
                    algorithm.chain_adapters = chain_adapters
                    algorithm.inputs = default_inputs
                    algorithm.dynamic_inputs = dynamic_inputs
                    continue
        return algorithm


    def _parse_inputs(self, nodes_list):
        """Read the inputs"""
        result = []
        for node in nodes_list:
            if node.nodeType != Node.ELEMENT_NODE or node.nodeName != ELEM_INPUT:
                continue
            input_ = self._read_all_attributes(node)
            req = node.getAttribute(ATT_REQUIRED)
            if req is not None and len(str(req)) > 0:
                input_[ATT_REQUIRED] = eval(req)
            if len(node.childNodes) > 0:
                for child_node in node.childNodes:
                    if child_node.nodeType != Node.ELEMENT_NODE:
                        continue
                    input_.update(self._read_all_attributes(child_node))
                    if child_node.nodeName == ELEM_NAME:
                        input_[ATT_NAME] = child_node.getAttribute(ATT_VALUE)
                    if child_node.nodeName == ELEM_LABEL:
                        input_[ATT_LABEL] = child_node.getAttribute(ATT_VALUE)
                    if child_node.nodeName == ELEM_DESCRIPTION:
                        input_[ATT_DESCRIPTION] = child_node.getAttribute(ATT_VALUE)
                    if child_node.nodeName == ELEM_TYPE:
                        self._parse_input_type(child_node, input_)
            result.append(input_)
        return result


    def _parse_input_type(self, child_node, input_):
        """
        Parse Type describing a particular Input.
        :param child_node is the original XML node which may contain minValue or maxValue attributes.
        :param input_ is the result dictionary, populated with some values already.
        """
        type_attrs = self._read_all_attributes(child_node)
        input_.update(type_attrs)
        input_[ATT_TYPE] = str(child_node.getAttribute(ATT_VALUE))
        default_value = child_node.getAttribute(ATT_DEFAULT)
        if default_value is not None and len(str(default_value)) > 0:
            input_[ATT_DEFAULT] = str(default_value)


    @classmethod
    def _read_all_attributes(cls, node):
        """From an XML node, return the map of all attributes."""
        atts = {}
        all_attributes = node.attributes
        if all_attributes is not None:
            for i in range(all_attributes.length):
                att = all_attributes.item(i)
                atts[att.name] = str(att.value)
        return atts


class PortletWrapper(object):
    """
    Transient model class, used for passing data when reading from XML into portlets memory
    """

    def __init__(self):
        self.name = None
        self.identifier = None
        self.inputs = []  # list of dictionaries
        self.chain_adapters = []
        self.dynamic_inputs = []


    def to_dict(self):
        """Convert into the expected UI tree"""
        alg = dict()
        alg[ATT_NAME] = self.name
        alg[ATT_IDENTIFIER] = self.identifier
        alg[ELEM_INPUTS] = self.get_inputs_dict()
        return alg


    def get_inputs_dict(self):
        """Return only inputs dictionary"""
        result = dict()
        for input_ in self.inputs:
            result[input_[ATT_NAME]] = input_
        return result
