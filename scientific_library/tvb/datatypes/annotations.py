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
                                ('region', 'i'),
                                ('relation', 'a'),
                                ('label', 'a'),
                                ('definition', 'a'),
                                ('url', 'a')])


class AnnotationTerm(object):
    """
    One single annotation (in the tree of annotations / region
    """

    def __init__(self, id, parent, region, relation, label, definition=None, url=None):
        self.id = id
        self.parent_id = parent
        self.region = region
        self.relation = relation
        self.label = label
        self.definition = definition
        self.url = url

    def to_numpy_row(self):
        return numpy.array([self.id, self.parent_id, self.region, self.relation, self.label, self.definition, self.url])


class AnnotationArray(Array):
    """
    Holds a flatten form for the annotations for a full connectivity.
    Each region in the connectivity can have None, or a tree of AnnotationTerms
    To be stored in a compound DS in H5.
    """

    dtype = types_basic.DType(default=ANNOTATION_DTYPE)


class ConnectivityAnnotations(MappedType):
    """
    Ontology annotations.
    """

    connectivity = connectivity.Connectivity

    region_annotations = AnnotationArray(
        default=numpy.array([], dtype=ANNOTATION_DTYPE),
        label="Region Annotations",
        doc="""Flat tree of annotations for every connectivity region.""")

    def set_annotations(self, annotation_terms):
        regions = numpy.array([], dtype=ANNOTATION_DTYPE)
        for ann in annotation_terms:
            regions = numpy.vstack([regions, ann.to_numpy_row()])
        self.region_annotations = regions
