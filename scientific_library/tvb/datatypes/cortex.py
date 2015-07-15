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
import numpy
import os
from tvb.basic.readers import try_get_absolute_path, FileReader
from tvb.datatypes.cortex_framework import CortexFramework
from tvb.datatypes.cortex_scientific import CortexScientific
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.surfaces import CorticalSurface


class Cortex(CortexScientific, CortexFramework, CorticalSurface):
    """
    This class brings together the scientific and framework methods that are
    associated with the Cortex dataType.

    ::

                             CortexData
                                 |
                                / \\
                 CortexFramework   CortexScientific
                                \ /
                                 |
                               Cortex


    """


    @classmethod
    def from_file(cls, source_file=os.path.join("cortex_reg13", "surface_cortex_reg13.zip"),
                  region_mapping_file=os.path.join("regionMapping_16k_76.txt"),
                  local_connectivity_file=None, eeg_projection_file=None, instance=None):

        result = super(Cortex, cls).from_file(source_file, instance)

        if instance is not None:
            # Called through constructor directly
            if result.region_mapping is None:
                result.region_mapping_data = RegionMapping.from_file()

            if not result.eeg_projection:
                result.eeg_projection = Cortex.from_file_projection_array()

            if result.local_connectivity is None:
                result.local_connectivity = LocalConnectivity.from_file()

        if region_mapping_file is not None:
            result.region_mapping_data = RegionMapping.from_file(region_mapping_file)

        if local_connectivity_file is not None:
            result.local_connectivity = LocalConnectivity.from_file(local_connectivity_file)

        if eeg_projection_file is not None:
            result.eeg_projection = Cortex.from_file_projection_array(eeg_projection_file)

        return result


    @staticmethod
    def from_file_projection_array(source_file="surface_reg_13_eeg_62.mat", matlab_data_name="ProjectionMatrix"):

        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        reader = FileReader(source_full_path)

        return reader.read_array(matlab_data_name=matlab_data_name)


    @staticmethod
    def from_file_region_mapping_array(source_file=os.path.join("cortex_reg13", "all_regions_cortex_reg13.txt")):

        source_full_path = try_get_absolute_path("tvb_data.surfaceData", source_file)
        reader = FileReader(source_full_path)

        return reader.read_array(dtype=numpy.int32)