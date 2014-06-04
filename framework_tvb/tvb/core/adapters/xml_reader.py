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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import xml.dom.minidom
from genxmlif import GenXmlIfError
from minixsv import pyxsval
from xml.dom.minidom import Node
from tvb.core.adapters.exceptions import XmlParserException
from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger
from tvb.basic.config.settings import TVBSettings


ATT_NAME = "name"
ATT_UI_NAME = "uiName"
ATT_ADDITIONAL_PATH = "additionalPath"
ATT_TYPE = "type"
ATT_SUB_SECTION = "subsection"
ATT_CODE = "code"
ATT_FILE_NAME = "matlab_file"
ATT_IMPORT = "import"
ATT_LABEL = "label"
ATT_DEFAULT = "default"
ATT_VALUE = "value"
ATT_FIELD = "field"
ATT_IDENTIFIER = "identifier"
ATT_ATTRIBUTES = "attributes"
ATT_REFERENCE = "reference"
ATT_DESCRIPTION = "description"
ATT_MINVALUE = "minValue"
ATT_MAXVALUE = "maxValue"
ATT_STEP = "step"
ATT_REQUIRED = "required"
ATT_QUATIFIER = 'quantifier'
ATT_ARRAY_ELEM_TYPE = 'elementType'
ATT_FILTER_VALUES = 'value_list'
ATT_PARAMETERS_PREFIX = 'parameters_prefix'

ELEM_OPTIONS = "options"
ELEM_OPTION = "option"
ELEM_ALGORITHM_GROUP = "algorithm_group"
ELEM_ALGORITHM = "algorithm"
ELEM_CODE = "code"
ELEM_FILE_NAME = "matlab_file"
ELEM_INPUTS = "inputs"
ELEM_INPUT = "input"
ELEM_NAME = "name"
ELEM_LABEL = "label"
ELEM_DESCRIPTION = "description"
ELEM_TYPE = "type"
ELEM_OUTPUTS = "outputs"
ELEM_OUTPUT = "output"
ELEM_FIELD = "field"
ELEM_CONDITIONS = "conditions"
ELEM_COND_FIELDS = "cond_fields"
ELEM_COND_OPS = "cond_operations"
ELEM_COND_VALUES = "cond_values"
ELEM_PRE_PROCESS = "pre_process"
ELEM_PYTHON_METHOD = "python_method"
ELEM_UI_METHOD = "ui_method"
ELEM_PARAMETERS = "parameters"
ELEM_PARAMETER = "parameter"

TYPE_INT = "int"
TYPE_STR = "str"
TYPE_FLOAT = "float"
TYPE_BOOL = "bool"
TYPE_DICT = "dict"
TYPE_SELECT = "select"
TYPE_UPLOAD = "upload"
TYPE_MULTIPLE = "selectMultiple"
TYPE_ARRAY = "array"
TYPE_DYNAMIC = "dynamic"
TYPE_LIST = "list"

QUANTIFIER_MANUAL = 'manual'
QUANTIFIER_UPLOAD = TYPE_UPLOAD
QUANTIFIER_FUNTION = 'function'

ALL_TYPES = (TYPE_STR, TYPE_FLOAT, TYPE_INT, TYPE_UPLOAD, TYPE_BOOL, TYPE_DICT,
             TYPE_ARRAY, TYPE_SELECT, TYPE_MULTIPLE, TYPE_DYNAMIC, TYPE_LIST)

ALGORITHMS_KEY = "algorithms"
INPUTS_KEY = "inputs"
OUTPUTS_KEY = "outputs"

### Global entity, caching already read XML files, for performance issues.
GLOBAL_LOADED_XML_READERS = {}


class XMLGroupReader():
    """
    Helper class to read from XML, a group of Algorithms definition.
    """
    
    @classmethod
    def get_instance(cls, xml_path):
        """
        Return a group reader for the given XML path. First check if a reader for this path
        is in memory, and if so return that instance. Otherwise create a new instance.
        """
        if xml_path in GLOBAL_LOADED_XML_READERS:
            return GLOBAL_LOADED_XML_READERS[xml_path]
        else:
            new_reader = cls(xml_path)
            GLOBAL_LOADED_XML_READERS[xml_path] = new_reader
            return new_reader
    
    def __init__(self, interface_file):
        """
        Validate and read XML.
        Private constructor. Should not be used directly, but rather through XMLGroupReader.get_instance
        """
        self.logger = get_logger(self.__class__.__module__ + '.')
        self.logger.debug("Starting to validate XML file " + interface_file)
        try:
            pyxsval.parseAndValidate(interface_file)
        except pyxsval.XsvalError, errstr:
            msg = "The XML file " + str(interface_file) + " is not valid. "
            self.logger.error(msg + "Error message: " + str(errstr))
            raise XmlParserException(msg + "Error message: " + str(errstr))
        except GenXmlIfError, errstr:
            msg = "The XML file " + str(interface_file) + " is not valid. "
            self.logger.error(msg + "Error message: " + str(errstr))
            raise XmlParserException(msg + "Error message: " + str(errstr))
        except IOError, _:
            self.logger.warning("Could not validate XML due to internet connection being down!")
            #self.logger.exception(excep)
            # Do not raise exception in this case

        self.logger.debug("Starting to parse XML file " + interface_file)
        doc_xml = xml.dom.minidom.parse(interface_file)
        tvb_node = doc_xml.lastChild
        root_node = None
        for child in tvb_node.childNodes:
            if child.nodeType == Node.ELEMENT_NODE and child.nodeName == ELEM_ALGORITHM_GROUP:
                root_node = child
                break
        if root_node is None:
            message = ("The given xml file is invalid. The file doesn't contain"
                       " a node with the name " + ELEM_ALGORITHM_GROUP)
            self.logger.error(message)
            raise XmlParserException(message)
        self._algorithm_group = None
        self.initialize(root_node)


    def initialize(self, root_node):
        """ Method to read from XML."""
        self._algorithm_group = self._parse_algorithm_group(root_node)


    @property
    def root_name(self):
        """Object name for the root element. To be used for re-building
        instances from introspection in case of not-groups. 
        Or as label for groups."""
        return self._algorithm_group[ATT_NAME]
    
    
    @property
    def subsection_name(self):
        """To be used to identity the group as CSS"""
        if ATT_SUB_SECTION in self._algorithm_group:
            return self._algorithm_group[ATT_SUB_SECTION]
        return self._algorithm_group[ATT_NAME]


    def get_group_name(self):
        """Return Algorithm Group Name"""
        return self._algorithm_group[ATT_NAME]
    
    
    def get_group_label(self):
        """Return UI label for Algorithm Group"""
        return self._algorithm_group[ATT_LABEL]


    def get_all_outputs(self):
        """
        Returns a list with all outputs for the current group. The result is a list of dictionaries.
        """
        result = []
        if self._algorithm_group is None or ALGORITHMS_KEY not in self._algorithm_group:
            return result
        algorithms = self._algorithm_group[ALGORITHMS_KEY]
        for key in algorithms.keys():
            alg = algorithms[key]
            result.extend(alg.outputs)
        return result


    def get_algorithms_dictionary(self):
        """ Retrieve the dictionary of available sub-Algorithm references."""
        result = dict()
        if self._algorithm_group is None or ALGORITHMS_KEY not in self._algorithm_group:
            return result
        algorithms = self._algorithm_group[ALGORITHMS_KEY]
        for key in algorithms.keys():
            result[key] = algorithms[key].to_dict()
        return result


    def get_matlab_file(self, algorithm_identifier):
        """
        The Matlab file where the code is located. Used to get the algorithm description. 
        """
        alg = self._get_algorithm(algorithm_identifier)
        if alg is None:
            return None
        return alg.file_name


    def get_code(self, algorithm_identifier):
        """Actual code to be executed for an algorithm.
        It can be PYTHON code, or MATLAB, for now."""
        alg = self._get_algorithm(algorithm_identifier)
        if alg is None:
            return None
        return alg.code


    def get_import(self, algorithm_identifier):
        """ Python import to be executed for an algorithm. optional"""
        alg = self._get_algorithm(algorithm_identifier)
        if alg is None:
            return None
        return alg.code_import


    def get_inputs(self, algorithm_identifier):
        """Return the interface Tree for a single sub-algorithm"""
        alg = self._get_algorithm(algorithm_identifier)
        if alg is None:
            return []
        return alg.inputs


    def get_outputs(self, algorithm_identifier):
        """ Return the list of outputs for a given algorithm"""
        alg = self._get_algorithm(algorithm_identifier)
        if alg is None:
            return []
        return alg.outputs


    def get_type(self):
        """Algorithm Class: Matlab/Python adapter."""
        return self._algorithm_group[ATT_TYPE]


    def get_ui_name(self):
        """UI display name"""
        ui_name = None
        if ATT_UI_NAME in self._algorithm_group.keys():
            ui_name = self._algorithm_group[ATT_UI_NAME]
        return ui_name
    
    
    def get_ui_description(self):
        """UI description"""
        result = ""
        if ATT_DESCRIPTION in self._algorithm_group.keys():
            result = self._algorithm_group[ATT_DESCRIPTION]
        return result


    def get_additional_path(self):
        """
        Extra file to be added to Matlab Path when executing algorithm.
        
        :returns: Absolute path for additional Matlab path.
        """
        additional_path = None
        if ATT_ADDITIONAL_PATH in self._algorithm_group.keys():
            additional_path = self._algorithm_group[ATT_ADDITIONAL_PATH]
            additional_path = os.path.join(TVBSettings.EXTERNALS_FOLDER_PARENT, additional_path)
        return additional_path


    def _get_algorithm(self, algorithm_identifier):
        """"Input tree for an algorithm."""
        if self._algorithm_group is None or ALGORITHMS_KEY not in self._algorithm_group:
            return None
        algorithms_dict = self._algorithm_group[ALGORITHMS_KEY]
        if algorithm_identifier not in algorithms_dict:
            return None
        return algorithms_dict[algorithm_identifier]


    def _parse_algorithm_group(self, node):
        """"Method which knows how to parse an algorithm group node"""
        if node.nodeType != Node.ELEMENT_NODE or node.nodeName != ELEM_ALGORITHM_GROUP:
            return None
        algorithm_group = XMLGroupReader._read_all_attributes(node)
        algorithms = {}
        if len(node.childNodes) > 0:
            for child in node.childNodes:
                if child.nodeName == ELEM_ALGORITHM:
                    alg = self._parse_algorithm(child)
                    if alg is not None:
                        algorithms[alg.identifier] = alg
            algorithm_group[ALGORITHMS_KEY] = algorithms
        return algorithm_group


    def _parse_algorithm(self, node):
        """"Method used for parsing an algorithm node"""
        if node.nodeType != Node.ELEMENT_NODE or node.nodeName != ELEM_ALGORITHM:
            return None

        algorithm = AlgorithmWrapper()
        algorithm.name = node.getAttribute(ATT_NAME)
        algorithm.identifier = node.getAttribute(ATT_IDENTIFIER)
        if len(node.childNodes) > 0:
            for child in node.childNodes:
                if child.nodeName == ELEM_CODE:
                    algorithm.code = child.getAttribute(ATT_VALUE)
                    algorithm.code_import = child.getAttribute(ATT_IMPORT)
                    continue
                if child.nodeName == ELEM_FILE_NAME:
                    algorithm.file_name = child.getAttribute(ATT_VALUE)
                    continue
                if child.nodeName == ELEM_INPUTS:
                    algorithm.inputs = self._parse_inputs(child.childNodes)
                    continue
                if child.nodeName == ELEM_OUTPUTS:
                    algorithm.outputs = self._parse_outputs(child.childNodes)
                    continue
        return algorithm


    def _parse_inputs(self, nodes_list):
        """Read the inputs"""
        result = []
        for node in nodes_list:
            if node.nodeType != Node.ELEMENT_NODE or node.nodeName != ELEM_INPUT:
                continue
            input_ = XMLGroupReader._read_all_attributes(node)
            req = node.getAttribute(ATT_REQUIRED)
            if req is not None and len(str(req)) > 0:
                input_[ATT_REQUIRED] = eval(req)
            if len(node.childNodes) > 0:
                for child_node in node.childNodes:
                    if child_node.nodeType != Node.ELEMENT_NODE:
                        continue
                    input_.update(XMLGroupReader._read_all_attributes(child_node))
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
        child_node is the original XML node.
            May contain minValue or maxValue attributes.
        input_ is the result dictionary, populated with some values already.
        """
        type_attrs = XMLGroupReader._read_all_attributes(child_node)
        input_.update(type_attrs)
        input_[ATT_TYPE] = str(child_node.getAttribute(ATT_VALUE))
        field_val = child_node.getAttribute(ATT_FIELD)
        if field_val is not None and len(str(field_val)) > 0:
            input_[ATT_FIELD] = str(field_val)
        default_value = child_node.getAttribute(ATT_DEFAULT)
        if default_value is not None and len(str(default_value)) > 0:
            input_[ATT_DEFAULT] = str(default_value)
        if len(child_node.childNodes) > 0:
            for type_child in child_node.childNodes:
                if ((input_[ATT_TYPE] == TYPE_SELECT or input_[ATT_TYPE] == TYPE_MULTIPLE) and
                        type_child.nodeType == Node.ELEMENT_NODE and type_child.nodeName == ELEM_OPTIONS):
                    ops = self._parse_options(type_child.childNodes)
                    input_[ELEM_OPTIONS] = ops
                if type_child.nodeName == ELEM_CONDITIONS:
                    filter_ = XMLGroupReader._get_filter(type_child.childNodes)
                    input_[ELEM_CONDITIONS] = filter_
                if type_child.nodeName == ELEM_PRE_PROCESS:
                    attr_dict = XMLGroupReader._parse_pre_process_element(type_child.childNodes)
                    input_.update(attr_dict)


    @classmethod
    def _parse_outputs(cls, nodes_list):
        """ Read output specifications"""
        result = []
        for node in nodes_list:
            if node.nodeType != Node.ELEMENT_NODE or node.nodeName != ELEM_OUTPUT:
                continue
            output = {ATT_TYPE: node.getAttribute(ATT_TYPE)}
            fields = []
            if len(node.childNodes) > 0:
                for field_n in node.childNodes:
                    if field_n.nodeType != Node.ELEMENT_NODE or field_n.nodeName != ELEM_FIELD:
                        continue
                    field = {ATT_NAME: field_n.getAttribute(ATT_NAME)}
                    field_val = field_n.getAttribute(ATT_VALUE)
                    if field_val is not None and len(str(field_val)) > 0:
                        field[ATT_VALUE] = str(field_val)
                    field[ATT_REFERENCE] = field_n.getAttribute(ATT_REFERENCE)
                    fields.append(field)
                output[ELEM_FIELD] = fields
            result.append(output)
        return result


    @classmethod
    def _parse_pre_process_element(cls, child_nodes):
        """
        Parse the child elements of a 'pre_process' element.
        """
        result = {}
        for node in child_nodes:
            if node.nodeType != Node.ELEMENT_NODE:
                continue
            if node.nodeName == ELEM_PYTHON_METHOD:
                result[ELEM_PYTHON_METHOD] = node.getAttribute(ATT_VALUE)
            if node.nodeName == ELEM_UI_METHOD:
                result[ELEM_UI_METHOD] = node.getAttribute(ATT_VALUE)
                result[ATT_PARAMETERS_PREFIX] = node.getAttribute(ATT_PARAMETERS_PREFIX)
            if node.nodeName == ELEM_PARAMETERS:
                params = {}
                if len(node.childNodes) > 0:
                    for child in node.childNodes:
                        if child.nodeType == Node.ELEMENT_NODE and child.nodeName == ELEM_PARAMETER:
                            params[child.getAttribute(ATT_NAME)] = XMLGroupReader._read_all_attributes(child)
                result[ELEM_PARAMETERS] = params
        return result


    def _parse_options(self, child_nodes):
        """Private - parse an XML nodes for Options"""
        result = []
        for node in child_nodes:
            if node.nodeType != Node.ELEMENT_NODE or node.nodeName != ELEM_OPTION:
                continue
            option = XMLGroupReader._read_all_attributes(node)
            option[ATT_VALUE] = node.getAttribute(ATT_VALUE)
            if len(node.childNodes) > 0:
                for child in node.childNodes:
                    if child.nodeType != Node.ELEMENT_NODE or child.nodeName != ELEM_INPUTS:
                        continue
                    option[ATT_ATTRIBUTES] = self._parse_inputs(child.childNodes)
                    break
            result.append(option)
        return result


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


    @classmethod            
    def _get_filter(cls, nodes_list):
        """Get default filter"""
        fields = None
        values = None
        operations = None
        for node in nodes_list:
            if node.nodeName == ELEM_COND_FIELDS:
                fields = eval(node.getAttribute(ATT_FILTER_VALUES))
            if node.nodeName == ELEM_COND_OPS:
                operations = eval(node.getAttribute(ATT_FILTER_VALUES))
            if node.nodeName == ELEM_COND_VALUES:
                values = eval(node.getAttribute(ATT_FILTER_VALUES))
        return FilterChain(fields=fields, values=values, operations=operations)


class AlgorithmWrapper:
    """Transient entity, for manipulating Algorithm related attributes"""

    def __init__(self):
        self.name = None
        self.identifier = None
        self.file_name = None
        self.code = None
        self.code_import = None
        self.inputs = []   # list of dictionaries
        self.outputs = []  # list of dictionaries

    def to_dict(self):
        """Convert into the expected UI tree"""
        alg = dict()
        alg[ATT_NAME] = self.name
        alg[ATT_IDENTIFIER] = self.identifier
        alg[ATT_CODE] = self.code
        alg[ATT_FILE_NAME] = self.file_name
        if self.code_import is not None:
            alg[ATT_IMPORT] = self.code_import
        alg[INPUTS_KEY] = self.get_inputs_dict()
        alg[OUTPUTS_KEY] = self.outputs
        return alg


    def get_inputs_dict(self):
        """Return only inputs dictionary"""
        result = dict()
        for input_ in self.inputs:
            result[input_[ATT_NAME]] = input_
        return result
    
    