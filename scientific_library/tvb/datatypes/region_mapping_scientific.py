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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
DataTypes for mapping some TVB DataTypes to a Connectivity (regions).

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

from tvb.datatypes.region_mapping_data import RegionMappingData, RegionVolumeMappingData


class RegionMappingScientific(RegionMappingData):
    """
    Scientific methods regarding RegionMapping DataType.
    """
    __tablename__ = None

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(RegionMappingScientific, self)._find_summary_info()
        summary.update({"Source Surface": self.surface.display_name,
                        "Source Surface GID": self.surface.gid,
                        "Connectivity GID": self.connectivity.gid,
                        "Connectivity": self.connectivity.display_name})
        return summary



class RegionVolumeMappingScientific(RegionVolumeMappingData):
    """
    Scientific methods regarding RegionVolumeMapping DataType.
    """
    __tablename__ = None


    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(RegionVolumeMappingScientific, self)._find_summary_info()
        summary.update({"Source Volume": self.volume.display_name,
                        "Source Volume GID": self.volume.gid,
                        "Connectivity GID": self.connectivity.gid,
                        "Connectivity": self.connectivity.display_name})
        return summary