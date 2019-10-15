# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
"""

import numpy
from tvb.datatypes.connectivity import Connectivity
from tvb.basic.neotraits.api import Attr, NArray, HasTraits, narray_summary_info

ANNOTATION_DTYPE = numpy.dtype([('id', 'i'),
                                ('parent_id', 'i'),
                                ('parent_left', 'i'),
                                ('parent_right', 'i'),
                                ('relation', 'S16'),
                                ('label', 'S128'),
                                ('definition', 'S1024'),
                                ('synonym', 'S2048'),
                                ('uri', 'S248'),
                                ('synonym_tvb_left', 'i'),
                                ('synonym_tvb_right', 'i')
                                ])

ICON_TVB = "/static/style/nodes/nodeRoot.png"
ICON_FOLDER = "/static/style/nodes/nodeFolder.png"
NODE_ID_TVB = "node_tvb_"
NODE_ID_TVB_ROOT = "node_tvb_root_"
NODE_ID_BRCO = "node_brco_"


class AnnotationTerm(object):
    """
    One single annotation node (in the tree of annotations / region)
    """

    def __init__(self, id, parent, parent_left, parent_right, relation, label, definition=None, synonym=None, uri=None,
                 tvb_left=None, tvb_right=None):
        self.id = id
        self.parent_id = parent
        self.parent_left = parent_left
        self.parent_right = parent_right
        self.relation = relation
        self.label = label
        self.definition = definition
        self.synonym = synonym
        self.uri = uri
        self.synonym_tvb_left = tvb_left or -1
        self.synonym_tvb_right = tvb_right or -1
        self.children = []

    def add_child(self, annotation_child):
        self.children.append(annotation_child)

    def to_tuple(self):
        return self.id, self.parent_id, self.parent_left, self.parent_right, \
               self.relation, self.label, self.definition, self.synonym, self.uri, \
               self.synonym_tvb_left, self.synonym_tvb_right

    def to_json(self, is_right_hemisphere=False, activation_patterns=None):

        children = []
        for child in self.children:
            children.append(child.to_json(is_right_hemisphere, activation_patterns))
        title = "URI: " + self.uri + "\n\nLabel: " + self.label + "\n\nDefinition: " + self.definition + \
                "\n\nSynonyms: " + self.synonym.replace("|", "\n")
        if activation_patterns is not None and str(self.id) in activation_patterns:
            connected_regions = activation_patterns[str(self.id)]
            title += "\n\nTVB " + str(len(connected_regions)) + " connected regions: " + str(connected_regions)

        if self.synonym_tvb_right >= 0 and self.synonym_tvb_left >= 0:
            # When TVB regions display differently
            synonym_id = self.synonym_tvb_right if is_right_hemisphere else self.synonym_tvb_left
            short_tvb_name = self.uri.split('#')[1]
            title = str(synonym_id) + " - " + short_tvb_name + "\n\n" + title
            return dict(data=dict(icon=ICON_TVB, title=short_tvb_name),
                        attr=dict(id=NODE_ID_TVB + str(synonym_id), title=title),
                        state="close", children=children)

        return dict(data=dict(icon=ICON_FOLDER, title=self.label.capitalize()),
                    attr=dict(id=NODE_ID_BRCO + str(self.id), title=title),
                    state="close", children=children)


class ConnectivityAnnotations(HasTraits):
    """
    Ontology annotations for a Connectivity.
    """

    connectivity = Attr(field_type=Connectivity)

    """
    Holds a flatten form for the annotations for a full connectivity.
    Each region in the connectivity can have None, or a tree of AnnotationTerms
    To be stored in a compound DS in H5.
    """
    region_annotations = NArray(
        default=numpy.array([], dtype=ANNOTATION_DTYPE),
        label="Region Annotations",
        doc="""Flat tree of annotations for every connectivity region.""")

    def set_annotations(self, annotation_terms):
        annotations = [ann.to_tuple() for ann in annotation_terms]
        annotations = numpy.array(annotations, dtype=ANNOTATION_DTYPE)
        self.region_annotations = annotations

    def summary_info(self):
        """
        Gather interesting summary information from an instance of this dataType.
        """
        summary = {"Connectivity": self.connectivity.display_name}
        summary.update(narray_summary_info(self.region_annotations, ar_name='region_annotations'))
        return summary

    def get_activation_patterns(self):
        """
        Group Annotation terms by URI.
        :return: map {brco_id: list of TVB regions IDs in which the same term is being subclass}
        """

        map_by_uri = {}
        for ann in self.region_annotations:
            ann_uri = ann[8]
            left, right = str(ann[2]), str(ann[3])
            if ann_uri not in map_by_uri:
                map_by_uri[ann_uri] = [left, right]
            else:
                if left not in map_by_uri[ann_uri]:
                    map_by_uri[ann_uri].append(left)
                if right not in map_by_uri[ann_uri]:
                    map_by_uri[ann_uri].append(right)

        map_by_brco_id = {}
        for ann in self.region_annotations:
            ann_uri = ann[8]
            ann_id = ann[0]
            map_by_brco_id[str(ann_id)] = map_by_uri[ann_uri]

        return map_by_brco_id

    def get_activation_pattern_labels(self):
        """
        :return: map {brco_id: list of TVB regions LABELS in which the same BRCO term is being subclass}
        """
        map_with_ids = self.get_activation_patterns()
        map_with_labels = dict()

        for ann_id, activated_ids in list(map_with_ids.items()):
            map_with_labels[ann_id] = []
            for string_idx in activated_ids:
                int_idx = int(string_idx)
                conn_label = self.connectivity.region_labels[int_idx]
                map_with_labels[ann_id].append(conn_label)

        return map_with_labels

    def tree_json(self):
        """
        :return: JSON to be rendered in a Tree of entities
        """
        annotations_map = dict()
        regions_map = dict()
        for i in range(self.connectivity.number_of_regions):
            regions_map[i] = []

        for ann in self.region_annotations:
            ann_obj = AnnotationTerm(ann[0], ann[1], ann[2], ann[3], ann[4], ann[5],
                                     ann[6], ann[7], ann[8], ann[9], ann[10])
            annotations_map[ann_obj.id] = ann_obj
            if ann_obj.parent_id < 0:
                # Root directly under a TVB region node
                regions_map[ann_obj.parent_left].append(ann_obj)
                regions_map[ann_obj.parent_right].append(ann_obj)
            elif ann_obj.parent_id in annotations_map:
                annotations_map[ann_obj.parent_id].add_child(ann_obj)
            else:
                self.logger.warn("Order of processing invalid parent %s child %s" % (ann_obj.parent_id, ann_obj.id))

        left_nodes, right_nodes = [], []
        activation_patterns = self.get_activation_pattern_labels()
        for region_idx, annotations_list in list(regions_map.items()):
            if_right_hemisphere = self.connectivity.is_right_hemisphere(region_idx)
            childred_json = []
            for ann_term in annotations_list:
                childred_json.append(ann_term.to_json(if_right_hemisphere, activation_patterns))
            # This node is built for every TVB region
            child_json = dict(data=dict(icon=ICON_TVB,
                                        title=self.connectivity.region_labels[region_idx]),
                              attr=dict(id=NODE_ID_TVB_ROOT + str(region_idx),
                                        title=str(region_idx) + " - " + self.connectivity.region_labels[region_idx]),
                              state="close", children=childred_json)
            if if_right_hemisphere:
                right_nodes.append(child_json)
            else:
                left_nodes.append(child_json)

        # Group everything under a single root
        left_root = dict(data=dict(title="Left Hemisphere", icon=ICON_FOLDER),
                         state="open", children=left_nodes)
        right_root = dict(data=dict(title="Right Hemisphere", icon=ICON_FOLDER),
                          state="open", children=right_nodes)
        root_root = dict(data=dict(title=self.display_name, icon=ICON_FOLDER),
                         state="open", children=[left_root, right_root])
        return root_root
