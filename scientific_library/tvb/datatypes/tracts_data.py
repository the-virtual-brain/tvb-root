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
module docstring
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from tvb.datatypes import arrays
from tvb.basic.traits import core
from tvb.basic.traits.types_mapped import MappedType
from tvb.datatypes.region_mapping import RegionVolumeMapping


class TractData(MappedType):
    vertices = arrays.PositionArray(
        label="Vertex positions",
        file_storage=core.FILE_STORAGE_EXPAND,
        order=-1,
        doc="""An array specifying coordinates for the tracts vertices.""")

    tract_start_idx = arrays.IntegerArray(
        label="Tract starting indices",
        order=-1,
        doc="""Where is the first vertex of a tract in the vertex array""")

    tract_region = arrays.IntegerArray(
        label="Tract region index",
        required=False,
        order=-1,
        doc="""
        An index used to find quickly all tract emerging from a region
        tract_region[i] is the region of the i'th tract. -1 represents the background
        """
    )

    region_volume_map = RegionVolumeMapping(
        label="Region volume Mapping used to create the tract_region index",
        required=False,
        order=-1
    )

    __generate_table__ = True

    @property
    def tracts_count(self):
        return len(self.tract_start_idx) - 1
