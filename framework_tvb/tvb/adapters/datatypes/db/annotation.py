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
"""

from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, ForeignKey, String
from tvb.adapters.datatypes.h5.annotation_h5 import ConnectivityAnnotations
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.core.entities.model.model_datatype import DataType

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


class ConnectivityAnnotationsIndex(DataType):
    """
    Ontology annotations for a Connectivity.
    """
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    connectivity_gid = Column(String(32), ForeignKey(ConnectivityIndex.gid), nullable=False)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_gid,
                                primaryjoin=ConnectivityIndex.gid == connectivity_gid, cascade='none')

    annotations_length = Column(Integer)

    def fill_from_has_traits(self, datatype):
        # type: (ConnectivityAnnotations)  -> None
        super(ConnectivityAnnotationsIndex, self).fill_from_has_traits(datatype)
        self.annotations_length = datatype.region_annotations.shape[0]
        self.connectivity_gid = datatype.connectivity.gid.hex
