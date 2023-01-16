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
"""

import xml.dom.minidom
from xml.dom.minidom import Node
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException
from tvb.adapters.datatypes.db.annotation import AnnotationTerm


class XMLParser(object):
    """
    Parser for XML with Connectivity annotations
    """
    LOGGER = get_logger(__name__)

    ATTR_TVB_ID = "id"
    ATTR_TVB_LABEL = "label"
    PREFIX_TVB_ID = "http://thevirtualbrain.org#"
    PREFIX_LEFT = ["l", "L", "left", "Left"]
    PREFIX_RIGHT = ["r", "R", "right", "Right"]

    ATTR_ONTOLOGY_URI = "uri"
    ATTR_ONTOLOGY_RELATION = "relation"
    ATTR_ONTOLOGY_LABEL = "label"
    ATTR_ONTOLOGY_SYN = "synonym"
    ATTR_ONTOLOGY_DEFINITION = "definition"

    def __init__(self, xml_file, region_labels):
        self.doc_xml = xml.dom.minidom.parse(xml_file)
        self.connectivity_labels = [rl.lower() for rl in region_labels]
        self.last_id = 0

    def _generate_new_id(self):
        self.last_id += 1
        return self.last_id

    def _find_region_idxs(self, region_label):
        """
        Find IDX in Connectivity based on label read from XML
        :param region_label: Label ar read from XML (including PREFIX_TVB_ID)
        :return: (left_hemisphere, right_hemisphere_idx)
        :raise: ParseException in case of not match
        """
        short_name = region_label.lower().replace(self.PREFIX_TVB_ID, "")

        left_idx = [self.connectivity_labels.index(prefix + short_name) for prefix in self.PREFIX_LEFT if
                    prefix + short_name in self.connectivity_labels]
        right_idx = [self.connectivity_labels.index(prefix + short_name) for prefix in self.PREFIX_RIGHT if
                     prefix + short_name in self.connectivity_labels]

        if len(left_idx) > 0 and len(right_idx) > 0:
            return left_idx[0], right_idx[0]

        raise ParseException("Could not match regionID '%s' from XML with the chosen connectivity" % region_label)

    def _parse_ontology_children(self, parent_node, parent_term_id, region_left, region_right):
        """
        To be called recursively, for processing OntologyTerm XML tags.
        """
        result = []
        for ont_node in parent_node.childNodes:
            if ont_node.nodeType != Node.ELEMENT_NODE:
                continue

            ont_uri = ont_node.getAttribute(self.ATTR_ONTOLOGY_URI)
            ont_relation = ont_node.getAttribute(self.ATTR_ONTOLOGY_RELATION)
            ont_label = ont_node.getAttribute(self.ATTR_ONTOLOGY_LABEL)
            ont_syn = ont_node.getAttribute(self.ATTR_ONTOLOGY_SYN)
            ont_definition = ont_node.getAttribute(self.ATTR_ONTOLOGY_DEFINITION)

            # if len(ont_syn) > 0 and ont_syn[:len(ont_syn) / 2] == ont_syn[len(ont_syn) / 2 + 1:]:
            #     ont_syn = ont_syn[:len(ont_syn) / 2]

            try:
                # Check if it is a TVB region
                syn_left, syn_right = self._find_region_idxs(ont_uri)
                ont_term = AnnotationTerm(self._generate_new_id(), parent_term_id, region_left, region_right,
                                          ont_relation, ont_label, ont_definition, ont_syn, ont_uri,
                                          syn_left, syn_right)
            except ParseException:

                ont_term = AnnotationTerm(self._generate_new_id(), parent_term_id, region_left, region_right,
                                          ont_relation, ont_label, ont_definition, ont_syn, ont_uri)
            # Call recursively for all child nodes
            child_terms = self._parse_ontology_children(ont_node, ont_term.id, region_left, region_right)
            result.append(ont_term)
            result.extend(child_terms)
        return result

    def read_annotation_terms(self):
        """
        Process XML file.
        :return List of AnnotationTerm instances.
        """
        result = []
        for tvb_node in self.doc_xml.lastChild.childNodes:
            if tvb_node.nodeType != Node.ELEMENT_NODE:
                continue

            tvb_id = tvb_node.getAttribute(self.ATTR_TVB_ID)
            tvb_label = tvb_node.getAttribute(self.ATTR_TVB_LABEL)
            self.LOGGER.debug("Processing region %s:%s" % (tvb_id, tvb_label))
            region_left, region_right = self._find_region_idxs(tvb_id)
            self.LOGGER.debug("Matched %s == (%d, %d)" % (tvb_label, region_left, region_right))
            ont_terms = self._parse_ontology_children(tvb_node, -1, region_left, region_right)
            result.extend(ont_terms)
        return result
