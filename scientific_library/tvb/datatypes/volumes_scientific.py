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
Scientific methods for the Volume datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import tvb.datatypes.volumes_data as volumes_data


class VolumeScientific(volumes_data.VolumeData):
    """ This class exists to add scientific methods to VolumeData. """
    __tablename__ = None
    
    
    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {"Volume type": self.__class__.__name__}
        summary["Origin"] = self.origin
        summary["Voxel size"] = self.voxel_size
        summary["Units"] = self.voxel_unit
        return summary


class ParcellationMaskScientific(volumes_data.ParcellationMaskData,
                                 VolumeScientific):
    """ This class exists to add scientific methods to ParcellationMaskData. """
    
    
    def _find_summary_info(self):
        """ Extend the base class's summary dictionary. """
        summary = super(ParcellationMaskScientific, self)._find_summary_info()
        summary["Volume shape"] = self.get_data_shape('data')
        summary["Number of regions"] = self.get_data_shape('region_labels')[0]
        return summary


class StructuralMRIScientific(volumes_data.StructuralMRIData,
                              VolumeScientific):
    """ This class exists to add scientific methods to StructuralMRIData. """
    pass

