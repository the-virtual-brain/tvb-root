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
The Sensors dataType. This brings together the scientific and framework 
methods that are associated with the sensor dataTypes.

.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

from tvb.datatypes import sensors_data
from tvb.datatypes import sensors_scientific
from tvb.datatypes import sensors_framework
from tvb.basic.readers import FileReader, try_get_absolute_path




class Sensors(sensors_scientific.SensorsScientific, sensors_framework.SensorsFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the Sensors DataType.
    
    ::
        
                            SensorsData
                                 |
                                / \\
                SensorsFramework   SensorsScientific
                                \ /
                                 |
                              Sensors
        
    
    """

    @classmethod
    def from_file(cls, source_file="EEG_unit_vectors_BrainProducts_62.txt.bz2", instance=None):

        if instance is None:
            result = cls()
        else:
            result = instance

        source_full_path = try_get_absolute_path("tvb_data.sensors", source_file)
        reader = FileReader(source_full_path)

        result.labels = reader.read_array(dtype="string", use_cols=(0,))
        result.locations = reader.read_array(use_cols=(1, 2, 3))

        return result



class SensorsEEG(sensors_scientific.SensorsEEGScientific,
                 sensors_framework.SensorsEEGFramework, Sensors):
    """
    This class brings together the scientific and framework methods that are
    associated with the SensorsEEG datatype.
    
    ::
        
                           SensorsEEGData
                                 |
                                / \\
             SensorsEEGFramework   SensorsEEGScientific
                                \ /
                                 |
                             SensorsEEG
        
    
    """

    __mapper_args__ = {'polymorphic_identity': sensors_data.EEG_POLYMORPHIC_IDENTITY}



class SensorsMEG(sensors_scientific.SensorsMEGScientific,
                 sensors_framework.SensorsMEGFramework, Sensors):
    """
    This class brings together the scientific and framework methods that are
    associated with the SensorsMEG datatype.
    
    ::
        
                           SensorsMEGData
                                 |
                                / \\
             SensorsMEGFramework   SensorsMEGScientific
                                \ /
                                 |
                             SensorsMEG
        
    
    """

    __mapper_args__ = {'polymorphic_identity': sensors_data.MEG_POLYMORPHIC_IDENTITY}


    @classmethod
    def from_file(cls, source_file="meg_channels_reg13.txt.bz2", instance=None):

        result = super(SensorsMEG, cls).from_file(source_file, instance)

        source_full_path = try_get_absolute_path("tvb_data.sensors", source_file)
        reader = FileReader(source_full_path)
        result.orientations = reader.read_array(use_cols=(4, 5, 6))

        return result



class SensorsInternal(sensors_scientific.SensorsInternalScientific,
                      sensors_framework.SensorsInternalFramework, Sensors):
    """
    This class brings together the scientific and framework methods that are
    associated with the SensorsInternal datatype.
    
    ::
        
                        SensorsInternalData
                                 |
                                / \\
        SensorsInternalFramework   SensorsInternalScientific
                                \ /
                                 |
                          SensorsInternal
        
    
    """

    __mapper_args__ = {'polymorphic_identity': sensors_data.INTERNAL_POLYMORPHIC_IDENTITY}


    @classmethod
    def from_file(cls, source_file="internal_39.txt.bz2", instance=None):
        return super(SensorsInternal, cls).from_file(source_file, instance)
