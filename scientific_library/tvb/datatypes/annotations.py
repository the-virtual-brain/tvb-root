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
"""

import numpy
from tvb.basic.traits import types_basic
from tvb.basic.traits.types_mapped import MappedType, Array
from tvb.datatypes import connectivity


ANNOTATION_DTYPE = numpy.dtype([('id', 'i'),
                                ('parent_id', 'i'),
                                ('region_left', 'i'),
                                ('region_right', 'i'),
                                ('relation', 'S16'),
                                ('label', 'S128'),
                                ('definition', 'S1024'),
                                ('synonym', 'S2048'),
                                ('uri', 'S248')])



class AnnotationTerm(object):
    """
    One single annotation node (in the tree of annotations / region)
    """


    def __init__(self, id, parent, region_left, region_right, relation, label, definition=None, synonym=None, uri=None):
        self.id = id
        self.parent_id = parent
        self.region_left = region_left
        self.region_right = region_right
        self.relation = relation
        self.label = label
        self.definition = definition
        self.synonym = synonym
        self.uri = uri
        self.children = []


    def add_child(self, annotation_child):
        self.children.append(annotation_child)


    def to_tuple(self):
        return self.id, self.parent_id, self.region_left, self.region_right, \
               self.relation, self.label, self.definition, self.synonym, self.uri


    def to_json(self):
        children = []
        for child in self.children:
            children.append(child.to_json())
        return dict(data=dict(id=str(self.id), title=self.label + " -- " + self.uri),
                    state="close", children=children)



class AnnotationArray(Array):
    """
    Holds a flatten form for the annotations for a full connectivity.
    Each region in the connectivity can have None, or a tree of AnnotationTerms
    To be stored in a compound DS in H5.
    """

    dtype = types_basic.DType(default=ANNOTATION_DTYPE)

    stored_metadata = [MappedType.METADATA_ARRAY_SHAPE]



class ConnectivityAnnotations(MappedType):
    """
    Ontology annotations for a Connectivity.
    """

    connectivity = connectivity.Connectivity

    region_annotations = AnnotationArray(
        default=numpy.array([], dtype=ANNOTATION_DTYPE),
        label="Region Annotations",
        doc="""Flat tree of annotations for every connectivity region.""")


    def set_annotations(self, annotation_terms):
        annotations = [ann.to_tuple() for ann in annotation_terms]
        annotations = numpy.array(annotations, dtype=ANNOTATION_DTYPE)
        self.region_annotations = annotations


    def _find_summary_info(self):
        """
        Gather interesting summary information from an instance of this dataType.
        """
        summary = {"Connectivity": self.connectivity.display_name}
        summary.update(self.get_info_about_array('region_annotations', [self.METADATA_ARRAY_SHAPE]))
        return summary


    def tree_json(self):
        """
        :return: JSON to be rendered in a Tree of entities
        """
        annotations_map = dict()
        regions_map = dict()
        for i in xrange(self.connectivity.number_of_regions):
            regions_map[i] = []

        for ann in self.region_annotations:
            ann_obj = AnnotationTerm(ann[0], ann[1], ann[2], ann[3], ann[4], ann[5], ann[6], ann[7], ann[8])
            annotations_map[ann_obj.id] = ann_obj
            if ann_obj.parent_id < 0:
                # Root directly under a TVB region node
                regions_map[ann_obj.region_left].append(ann_obj)
                regions_map[ann_obj.region_right].append(ann_obj)
            elif ann_obj.parent_id in annotations_map:
                annotations_map[ann_obj.parent_id].add_child(ann_obj)
            else:
                self.logger.warn("Order of processing invalid parent %s child %s" % (ann_obj.parent_id, ann_obj.id))

        trees = []
        for region_idx, annotations_list in regions_map.iteritems():
            childred_json = []
            for ann_term in annotations_list:
                childred_json.append(ann_term.to_json())
            # This node is built for every TVB region
            trees.append(dict(data=dict(title=self.connectivity.region_labels[region_idx],
                                        icon="/static/style/nodes/nodeRoot.png"),
                              state="open",
                              children=childred_json))
        # Group everything under a single root
        result = dict(data=dict(title=self.display_name,
                                icon="/static/style/nodes/nodeRoot.png"),
                      state="open",
                      children=trees)
        return result
