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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

from xml.dom.minidom import Node
import tvb.core.adapters.xml_reader as xml_reader

KEY_DYNAMIC = 'dynamic'
KEY_STATIC = 'static'
KEY_FIELD = 'field'
 
ATT_MODULE = "module"
ATT_CHAIN = 'chain'
ATT_OVERWRITE = 'overwrite'


class XMLPortletReader(xml_reader.XMLGroupReader):
    """
    Helper class to read from XML, a group of Portlets definition. Extend
    the functionality of XMLGroupReader with any specific requirements.
    """    
        
    def get_adapters_chain(self, algorithm_identifier):
        """ Return the list of outputs for a given algorithm"""
        alg = self._get_algorithm(algorithm_identifier)
        if alg is None:
            return []
        return alg.chain_adapters
    
    def get_dynamic_inputs(self, algorithm_identifier):
        """Return a list with all dynamic attribute names for current algorithm."""
        alg = self._get_algorithm(algorithm_identifier)
        if alg is None:
            return []
        return alg.dynamic_inputs
        
    def _parse_algorithm(self, node):
        """"Method used for parsing an algorithm node"""
        if node.nodeType != Node.ELEMENT_NODE or node.nodeName != xml_reader.ELEM_ALGORITHM:
            return None

        algorithm = PortletWrapper()
        algorithm.name = node.getAttribute(xml_reader.ATT_NAME)
        algorithm.identifier = node.getAttribute(xml_reader.ATT_IDENTIFIER)
        if len(node.childNodes) > 0:
            for child in node.childNodes:
                if child.nodeName == xml_reader.ELEM_CODE:
                    algorithm.code = child.getAttribute(xml_reader.ATT_VALUE)
                    algorithm.code_import = child.getAttribute(xml_reader.ATT_IMPORT)
                    continue
                if child.nodeName == xml_reader.ELEM_INPUTS:
                    all_inputs = self._parse_inputs(child.childNodes)
                    chain_adapters = []
                    dynamic_inputs = []
                    default_inputs = []
                    for one_input in all_inputs:
                        if one_input[xml_reader.ATT_NAME].startswith(ATT_CHAIN):
                            chain_adapters.append(one_input)
                        elif one_input[ATT_OVERWRITE] in one_input:
                            dynamic_inputs.append(one_input)
                        else:
                            default_inputs.append(one_input)
                    algorithm.chain_adapters = chain_adapters
                    algorithm.inputs = default_inputs
                    algorithm.dynamic_inputs = dynamic_inputs
                    continue
                if child.nodeName == xml_reader.ELEM_OUTPUTS:
                    algorithm.outputs = self._parse_outputs(child.childNodes)
                    continue
        return algorithm
    

class PortletWrapper(xml_reader.AlgorithmWrapper):
    """Extend with new required attributes for a portlet"""
    
    def __init__(self):
        xml_reader.AlgorithmWrapper.__init__(self)
        self.chain_adapters = []
        self.inputs = []
        self.dynamic_inputs = []

