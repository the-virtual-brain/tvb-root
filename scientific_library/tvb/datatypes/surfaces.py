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
Surface relates DataTypes. This brings together the scientific and framework 
methods that are associated with the surfaces data.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

import os
import numpy

from tvb.datatypes import surfaces_scientific
from tvb.datatypes import surfaces_framework
from tvb.datatypes import surfaces_data
from tvb.basic.readers import ZipReader, try_get_absolute_path
from tvb.datatypes.surfaces_data import WHITE_MATTER


CORTICAL = surfaces_data.CORTICAL
OUTER_SKIN = surfaces_data.OUTER_SKIN
INNER_SKULL = surfaces_data.INNER_SKULL
OUTER_SKULL = surfaces_data.OUTER_SKULL
EEG_CAP = surfaces_data.EEG_CAP
FACE = surfaces_data.FACE



class Surface(surfaces_scientific.SurfaceScientific, surfaces_framework.SurfaceFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the Surface DataType.
    
    ::
        
                            SurfaceData
                                 |
                                / \\
                SurfaceFramework   SurfaceScientific
                                \ /
                                 |
                              Surface
        
    
    """

    @classmethod
    def from_file(cls, source_file="cortex_16384.zip", instance=None):
        """
        Construct a Surface from source_file.
        """

        if instance is None:
            result = cls()
        else:
            result = instance

        source_full_path = try_get_absolute_path("tvb_data.surfaceData", source_file)
        reader = ZipReader(source_full_path)

        result.vertices = reader.read_array_from_file("vertices.txt")
        result.vertex_normals = reader.read_array_from_file("normals.txt")
        result.triangles = reader.read_array_from_file("triangles.txt", dtype=numpy.int32)

        return result


    def configure(self):
        """
        Make sure both Scientific and Framework configure methods are called.
        """
        surfaces_scientific.SurfaceScientific.configure(self)
        surfaces_framework.SurfaceFramework.configure(self)


    def validate(self):
        """
        Combines scientific and framework surface validations.
        """
        result_sci = surfaces_scientific.SurfaceScientific.validate(self)
        result_fr = surfaces_framework.SurfaceFramework.validate(self)

        validation_result = result_sci.merge(result_fr)

        self.user_tag_3 = validation_result.summary()
        return validation_result



class WhiteMatterSurface(surfaces_scientific.WhiteMatterSurfaceScientific,
                      surfaces_framework.WhiteMatterSurfaceFramework, Surface):
    """
    This class brings together the scientific and framework methods that are
    associated with the WhiteMatterSurface DataType.

    ::

                        WhiteMatterSurfaceData
                                 |
                                / \\
        WhiteMatterSurfaceFramework   WhiteMatterSurfaceScientific
                                \ /
                                 |
                          WhiteMatterSurface


    """
    __mapper_args__ = {'polymorphic_identity': WHITE_MATTER}


class CorticalSurface(surfaces_scientific.CorticalSurfaceScientific,
                      surfaces_framework.CorticalSurfaceFramework, Surface):
    """
    This class brings together the scientific and framework methods that are
    associated with the CorticalSurface DataType.
    
    ::
        
                        CorticalSurfaceData
                                 |
                                / \\
        CorticalSurfaceFramework   CorticalSurfaceScientific
                                \ /
                                 |
                          CorticalSurface
        
    
    """
    __mapper_args__ = {'polymorphic_identity': CORTICAL}



class SkinAir(surfaces_scientific.SkinAirScientific, surfaces_framework.SkinAirFramework, Surface):
    """
    This class brings together the scientific and framework methods that are
    associated with the SkinAir DataType.
    
    ::
        
                            SkinAirData
                                 |
                                / \\
                SkinAirFramework   SkinAirScientific
                                \ /
                                 |
                              SkinAir
        
    
    """
    __mapper_args__ = {'polymorphic_identity': OUTER_SKIN}

    @classmethod
    def from_file(cls, source_file="outer_skin_4096.zip", instance=None):
        return super(SkinAir, cls).from_file(source_file, instance)



class BrainSkull(surfaces_scientific.BrainSkullScientific, surfaces_framework.BrainSkullFramework, Surface):
    """ 
    This class brings together the scientific and framework methods that are
    associated with the BrainSkull dataType.
    
    ::
        
                           BrainSkullData
                                 |
                                / \\
             BrainSkullFramework   BrainSkullScientific
                                \ /
                                 |
                             BrainSkull
        
    
    """
    __mapper_args__ = {'polymorphic_identity': INNER_SKULL}

    @classmethod
    def from_file(cls, source_file="inner_skull_4096.zip", instance=None):
        return super(BrainSkull, cls).from_file(source_file, instance)



class SkullSkin(surfaces_scientific.SkullSkinScientific, surfaces_framework.SkullSkinFramework, Surface):
    """
    This class brings together the scientific and framework methods that are
    associated with the SkullSkin dataType.
    
    ::
        
                           SkullSkinData
                                 |
                                / \\
              SkullSkinFramework   SkullSkinScientific
                                \ /
                                 |
                             SkullSkin
        
    
    """
    __mapper_args__ = {'polymorphic_identity': OUTER_SKULL}

    @classmethod
    def from_file(cls, source_file="outer_skull_4096.zip", instance=None):
        return super(SkullSkin, cls).from_file(source_file, instance)


##--------------------- CLOSE SURFACES End Here---------------------------------------##


##--------------------- OPEN SURFACES Start Here---------------------------------------##


class OpenSurface(surfaces_scientific.OpenSurfaceScientific, surfaces_framework.OpenSurfaceFramework, Surface):
    """ 
    This class brings together the scientific and framework methods that are
    associated with the OpenSurface dataType.
    
    ::
        
                           OpenSurfaceData
                                 |
                                / \\
             OpenSurfaceFramework   OpenSurfaceScientific
                                \ /
                                 |
                             OpenSurface
        
    
    """
    pass



class EEGCap(surfaces_scientific.EEGCapScientific, surfaces_framework.EEGCapFramework, OpenSurface):
    """ 
    This class brings together the scientific and framework methods that are
    associated with the EEGCap dataType.
    
    ::
        
                           EEGCapData
                                 |
                                / \\
             EEGCapFramework   EEGCapScientific
                                \ /
                                 |
                             EEGCap
        
    
    """
    __mapper_args__ = {'polymorphic_identity': EEG_CAP}

    @classmethod
    def from_file(cls, source_file="scalp_1082.zip", instance=None):
        return super(EEGCap, cls).from_file(source_file, instance)



class FaceSurface(surfaces_scientific.FaceSurfaceScientific, surfaces_framework.FaceSurfaceFramework, OpenSurface):
    """ 
    This class brings together the scientific and framework methods that are
    associated with the FaceSurface dataType.
    
    ::
        
                           FaceSurfaceData
                                 |
                                / \\
             FaceSurfaceFramework   FaceSurfaceScientific
                                \ /
                                 |
                             FaceSurface
        
    
    """
    __mapper_args__ = {'polymorphic_identity': FACE}

    @classmethod
    def from_file(cls, source_file="face_8614.zip", instance=None):
        return super(FaceSurface, cls).from_file(source_file, instance)

##--------------------- OPEN SURFACES End Here---------------------------------------##


##--------------------- SURFACES ADJIACENT classes start Here---------------------------------------##


def make_surface(surface_type):
    """
    Build a Surface instance, based on an input type
    :param surface_type: one of the supported surface types
    :return: Instance of the corresponding surface lass, or None
    """
    if surface_type in [CORTICAL, "Pial"] or surface_type.startswith("Cortex"):
        return CorticalSurface()
    elif surface_type == INNER_SKULL:
        return BrainSkull()
    elif surface_type == OUTER_SKULL:
        return SkullSkin()
    elif surface_type in [OUTER_SKIN, "SkinAir"]:
        return SkinAir()
    elif surface_type == EEG_CAP:
        return EEGCap()
    elif surface_type == FACE:
        return FaceSurface()
    elif surface_type == WHITE_MATTER:
        return WhiteMatterSurface()

    return None


def center_vertices(vertices):
    """
    Centres the vertices using means along axes.
    :param vertices: a numpy array of shape (n, 3)
    :returns: the centered array
    """
    return vertices - numpy.mean(vertices, axis=0).reshape((1, 3))


ALL_SURFACES_SELECTION = [{'name': 'Cortical Surface', 'value': CORTICAL},
                          {'name': 'Brain Skull', 'value': INNER_SKULL},
                          {'name': 'Skull Skin', 'value': OUTER_SKULL},
                          {'name': 'Skin Air', 'value': OUTER_SKIN},
                          {'name': 'EEG Cap', 'value': EEG_CAP},
                          {'name': 'Face Surface', 'value': FACE},
                          {'name': 'White Matter Surface', 'value':WHITE_MATTER}]