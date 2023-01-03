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

import numpy
from tvb.basic.neotraits.api import NArray, Attr, HasTraits
from tvb.core.neotraits.h5 import H5File, DataSet, Reference, STORE_STRING
from tvb.datatypes.connectivity import Connectivity

ANNOTATION_DTYPE = numpy.dtype([('id', 'i'),
                                ('parent_id', 'i'),
                                ('parent_left', 'i'),
                                ('parent_right', 'i'),
                                ('relation', STORE_STRING),  # S16
                                ('label', STORE_STRING),  # S256
                                ('definition', STORE_STRING),  # S1024
                                ('synonym', STORE_STRING),  # S2048
                                ('uri', STORE_STRING),  # S256
                                ('synonym_tvb_left', 'i'),
                                ('synonym_tvb_right', 'i')
                                ])


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
        default=numpy.array([], dtype=ANNOTATION_DTYPE), dtype=ANNOTATION_DTYPE,
        label="Region Annotations",
        doc="""Flat tree of annotations for every connectivity region.""")

    def set_annotations(self, annotation_terms):
        annotations = [ann.to_tuple() for ann in annotation_terms]
        annotations = numpy.array(annotations, dtype=ANNOTATION_DTYPE)
        self.region_annotations = annotations

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


class ConnectivityAnnotationsH5(H5File):
    """
    Ontology annotations for a Connectivity.
    """

    def __init__(self, path):
        super(ConnectivityAnnotationsH5, self).__init__(path)
        self.region_annotations = DataSet(ConnectivityAnnotations.region_annotations, self)
        self.connectivity = Reference(ConnectivityAnnotations.connectivity, self)

    def store(self, datatype, scalars_only=False, store_references=True):
        # type: (ConnectivityAnnotations, bool, bool) -> None
        super(ConnectivityAnnotationsH5, self).store(datatype, scalars_only, store_references)
