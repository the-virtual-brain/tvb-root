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
The Data component of Sensors datatypes, for the moment just EEG and MEG,
however, ECoG, depth electrodes, etc should be supported...

Sensors uses:
    locations and labels for visualisation
    combined with source and head surfaces to generate projection matrices used
    in monitors such as EEG, MEG...

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

from tvb.basic.traits.types_mapped import MappedType
import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays


EEG_POLYMORPHIC_IDENTITY = "EEG"
MEG_POLYMORPHIC_IDENTITY = "MEG"
INTERNAL_POLYMORPHIC_IDENTITY = "Internal"



class SensorsData(MappedType):
    """
    Base Sensors class.
    All sensors have locations. 
    Some will have orientations, e.g. MEG.
    """

    _ui_name = "Unknown sensors"

    sensors_type = basic.String

    __mapper_args__ = {'polymorphic_on': 'sensors_type'}

    labels = arrays.StringArray(
        label="Sensor labels")

    locations = arrays.PositionArray(
        label="Sensor locations")

    has_orientation = basic.Bool(default=False)

    orientations = arrays.OrientationArray(required=False)

    number_of_sensors = basic.Integer(
        label="Number of sensors",
        doc="""The number of sensors described by these Sensors.""")



class SensorsEEGData(SensorsData):
    """
    EEG sensor locations are represented as unit vectors, these need to be
    combined with a head(outer-skin) surface to obtain actual sensor locations
    ::
        
                              position
                                 |
                                / \\
                               /   \\
        file columns: labels, x, y, z
        
    """
    _ui_name = "EEG Sensors"

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': EEG_POLYMORPHIC_IDENTITY}

    sensors_type = basic.String(default=EEG_POLYMORPHIC_IDENTITY)

    has_orientation = basic.Bool(default=False, order=-1)




class SensorsMEGData(SensorsData):
    """
    These are actually just SQUIDS. Axial or planar gradiometers are achieved
    by calculating lead fields for two sets of sensors and then subtracting...
    ::
        
                              position  orientation
                                 |           |
                                / \         / \\
                               /   \       /   \\
        file columns: labels, x, y, z,   dx, dy, dz
        
    """
    _ui_name = "MEG sensors"

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': MEG_POLYMORPHIC_IDENTITY}

    sensors_type = basic.String(default=MEG_POLYMORPHIC_IDENTITY)

    orientations = arrays.OrientationArray(
        label="Sensor orientations",
        doc="An array representing the orientation of the MEG SQUIDs")

    has_orientation = basic.Bool(default=True, order=-1)




class SensorsInternalData(SensorsData):
    """
    Sensors inside the brain...
    """
    _ui_name = "Internal Sensors"

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': INTERNAL_POLYMORPHIC_IDENTITY}

    sensors_type = basic.String(default=INTERNAL_POLYMORPHIC_IDENTITY)

