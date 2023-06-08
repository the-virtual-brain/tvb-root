# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.datatypes.region_mapping import RegionMapping as TVBRegionMapping
from tvb.datatypes.region_mapping import RegionVolumeMapping as TVBRegionVolumeMapping


class RegionMapping(TVBRegionMapping, BaseModel):

    def to_tvb_instance(self, datatype=TVBRegionMapping, **kwargs):
        return super(RegionMapping, self).to_tvb_instance(datatype, **kwargs)


class CorticalRegionMapping(RegionMapping):
    pass


class SubcorticalRegionMapping(RegionMapping):
    pass


class RegionVolumeMapping(TVBRegionVolumeMapping, BaseModel):

    def to_tvb_instance(self, **kwargs):
        return super(RegionVolumeMapping, self).to_tvb_instance(TVBRegionVolumeMapping, **kwargs)