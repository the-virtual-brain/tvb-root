# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

The Data component of Surfaces DataTypes.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.types_mapped import MappedType, SparseMatrix
from tvb.basic.traits.core import FILE_STORAGE_NONE


LOG = get_logger(__name__)

OUTER_SKIN = "Skin Air"
OUTER_SKULL = "Skull Skin"
INNER_SKULL = "Brain Skull"
CORTICAL = "Cortical Surface"
WHITE_MATTER = "White Matter"
EEG_CAP = "EEG Cap"
FACE = "Face"


##--------------------- CLOSE SURFACES Start Here---------------------------------------##

class SurfaceData(MappedType):
    """
    This class primarily exists to bundle the structural Surface data into a 
    single object.
    """

    vertices = arrays.PositionArray(
        label="Vertex positions",
        order=-1,
        doc="""An array specifying coordinates for the surface vertices.""")

    triangles = arrays.IndexArray(
        label="Triangles",
        order=-1,
        target=vertices,
        doc="""Array of indices into the vertices, specifying the triangles which define the surface.""")

    vertex_normals = arrays.OrientationArray(
        label="Vertex normal vectors",
        order=-1,
        doc="""An array of unit normal vectors for the surfaces vertices.""")

    triangle_normals = arrays.OrientationArray(
        label="Triangle normal vectors",
        order=-1,
        doc="""An array of unit normal vectors for the surfaces triangles.""")

    geodesic_distance_matrix = SparseMatrix(
        label="Geodesic distance matrix",
        order=-1,
        required=False,
        file_storage=FILE_STORAGE_NONE,
        doc="""A sparse matrix of truncated geodesic distances""")  # 'CS'

    number_of_vertices = basic.Integer(
        label="Number of vertices",
        order=-1,
        doc="""The number of vertices making up this surface.""")

    number_of_triangles = basic.Integer(
        label="Number of triangles",
        order=-1,
        doc="""The number of triangles making up this surface.""")

    edge_mean_length = basic.Float(order=-1)

    edge_min_length = basic.Float(order=-1)

    edge_max_length = basic.Float(order=-1)

    ##--------------------- FRAMEWORK ATTRIBUTES -----------------------------##

    hemisphere_mask = arrays.BoolArray(
        label="An array specifying if a vertex belongs to the right hemisphere",
        file_storage=FILE_STORAGE_NONE,
        required=False,
        order=-1)

    zero_based_triangles = basic.Bool(order=-1)

    split_triangles = arrays.IndexArray(order=-1, required=False)

    number_of_split_slices = basic.Integer(order=-1)

    split_slices = basic.Dict(order=-1)

    bi_hemispheric = basic.Bool(order=-1)

    surface_type = basic.String

    valid_for_simulations = basic.Bool(order=-1)

    __mapper_args__ = {'polymorphic_on': 'surface_type'}


class WhiteMatterSurfaceData(SurfaceData):
    """
    The boundary between the white and gray matter
    """

    _ui_name = "A white matter surface"

    surface_type = basic.String(default=WHITE_MATTER)

    ##--------------------- FRAMEWORK ATTRIBUTES -----------------------------##
    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': WHITE_MATTER}


class CorticalSurfaceData(SurfaceData):
    """
    A surface for describing the Brain Cortical area.
    """

    _ui_name = "A cortical surface"

    surface_type = basic.String(default=CORTICAL, order=-1)

    ##--------------------- FRAMEWORK ATTRIBUTES -----------------------------##
    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': CORTICAL}



class SkinAirData(SurfaceData):
    """
    A surface defining the boundary between the skin and the air.
    """

    _ui_name = "Skin"

    surface_type = basic.String(default=OUTER_SKIN)

    ##--------------------- FRAMEWORK ATTRIBUTES -----------------------------##
    __mapper_args__ = {'polymorphic_identity': OUTER_SKIN}

    __generate_table__ = True



class BrainSkullData(SurfaceData):
    """
    A surface defining the boundary between the brain and the skull.
    """

    _ui_name = "Inside of the skull"

    surface_type = basic.String(default=INNER_SKULL)

    ##--------------------- FRAMEWORK ATTRIBUTES -----------------------------##
    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': INNER_SKULL}



class SkullSkinData(SurfaceData):
    """
    A surface defining the boundary between the skull and the skin.
    """

    _ui_name = "Outside of the skull"

    surface_type = basic.String(default=OUTER_SKULL)


    ##--------------------- FRAMEWORK ATTRIBUTES -----------------------------##
    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': OUTER_SKULL}



##--------------------- CLOSE SURFACES End Here---------------------------------------##

##--------------------- OPEN SURFACES Start Here---------------------------------------##

class OpenSurfaceData(SurfaceData):
    """
    A base class for all surfaces that are open (eg. CapEEG or Face Surface).
    """
    __tablename__ = None



class EEGCapData(OpenSurfaceData):
    """
    A surface defining the EEG Cap.
    """
    _ui_name = "EEG Cap"

    surface_type = basic.String(default=EEG_CAP)

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': EEG_CAP}



class FaceSurfaceData(OpenSurfaceData):
    """
    A surface defining the face of a human.
    """
    _ui_name = "Face Surface"

    surface_type = basic.String(default=FACE)

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': FACE}


##--------------------- OPEN SURFACES End Here---------------------------------------##

##--------------------- SURFACES ADJIACENT classes start Here---------------------------------------##


class ValidationResult(object):
    """
    Used by surface validate methods to report non-fatal failed validations
    """
    def __init__(self):
        self.warnings = []

    def add_warning(self, msg, data):
        self.warnings.append((msg, data))
        self._log(msg, data)

    def _log(self, msg, data):
        LOG.warn(msg)
        if data:
            LOG.debug(data)

    def merge(self, other):
        r = ValidationResult()
        r.warnings = self.warnings + other.warnings
        return r

    def summary(self):
        return '  |  '.join(msg for msg, _ in self.warnings)