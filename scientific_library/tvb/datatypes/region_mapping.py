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

import numpy
import os
from tvb.basic.readers import try_get_absolute_path, FileReader
from tvb.datatypes.region_mapping_framework import RegionMappingFramework, RegionVolumeMappingFramework
from tvb.datatypes.region_mapping_scientific import RegionMappingScientific, RegionVolumeMappingScientific



class RegionMapping(RegionMappingFramework, RegionMappingScientific):
    """
    This class brings together the scientific and framework methods that are
    associated with the RegionMapping dataType.

    ::

                        RegionMappingData
                                 |
                                / \\
          RegionMappingFramework   RegionMappingScientific
                                \ /
                                 |
                          RegionMapping


    """

    @staticmethod
    def from_file(source_file=os.path.join("cortex_reg13", "region_mapping",
                                           "o52r00_irp2008_hemisphere_both_subcortical_false_regions_74.txt.bz2"),
                  instance=None):

        if instance is None:
            result = RegionMapping()
        else:
            result = instance

        source_full_path = try_get_absolute_path("tvb_data.surfaceData", source_file)
        reader = FileReader(source_full_path)

        result.array_data = reader.read_array(dtype=numpy.int32)
        return result



class RegionVolumeMapping(RegionVolumeMappingFramework, RegionVolumeMappingScientific):
    """
    This class brings together the scientific and framework methods that are
    associated with the RegionMapping dataType.

    ::

                        RegionVolumeMappingData
                                 |
                                / \\
    RegionVolumeMappingFramework   RegionVolumeMappingScientific
                                \ /
                                 |
                          RegionVolumeMapping


    """
