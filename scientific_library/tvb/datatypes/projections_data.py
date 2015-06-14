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
The Data component of ProjectionMatrices DataTypes.
"""

import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays
from tvb.datatypes import surfaces, sensors
from tvb.basic.traits.types_mapped import MappedType

EEG_POLYMORPHIC_IDENTITY = "projEEG"
MEG_POLYMORPHIC_IDENTITY = "projMEG"
SEEG_POLYMORPHIC_IDENTITY = "projSEEG"



class ProjectionData(MappedType):
    """
    Base DataType for representing a ProjectionMatrix.
    The projection is between a source of type CorticalSurface and a set of Sensors.
    """

    projection_type = basic.String

    __mapper_args__ = {'polymorphic_on': 'projection_type'}

    brain_skull = surfaces.BrainSkull(label="Brain Skull", default=None, required=False,
                                      doc="""Boundary between skull and cortex domains.""")

    skull_skin = surfaces.SkullSkin(label="Skull Skin", default=None, required=False,
                                    doc="""Boundary between skull and skin domains.""")

    skin_air = surfaces.SkinAir(label="Skin Air", default=None, required=False,
                                doc="""Boundary between skin and air domains.""")

    conductances = basic.Dict(label="Domain conductances", required=False,
                              default={'air': 0.0, 'skin': 1.0, 'skull': 0.01, 'brain': 1.0},
                              doc=""" A dictionary representing the conductances of ... """)

    sources = surfaces.CorticalSurface(label="surface or region", default=None, required=True)

    sensors = sensors.Sensors(label="Sensors", default=None, required=False,
                              doc=""" A set of sensors to compute projection matrix for them. """)

    projection_data = arrays.FloatArray(label="Projection Matrix Data", default=None, required=True)



class ProjectionSurfaceEEGData(ProjectionData):
    """
    Specific projection, from a CorticalSurface to EEG sensors.
    """

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': EEG_POLYMORPHIC_IDENTITY}

    projection_type = basic.String(default=EEG_POLYMORPHIC_IDENTITY)

    sensors = sensors.SensorsEEG



class ProjectionSurfaceMEGData(ProjectionData):
    """
    Specific projection, from a CorticalSurface to MEG sensors.
    """

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': MEG_POLYMORPHIC_IDENTITY}

    projection_type = basic.String(default=MEG_POLYMORPHIC_IDENTITY)

    sensors = sensors.SensorsMEG



class ProjectionSurfaceSEEGData(ProjectionData):
    """
    Specific projection, from a CorticalSurface to SEEG sensors.
    """

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': SEEG_POLYMORPHIC_IDENTITY}

    projection_type = basic.String(default=SEEG_POLYMORPHIC_IDENTITY)

    sensors = sensors.SensorsInternal


    
    
    
    
    