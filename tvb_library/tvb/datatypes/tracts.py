# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
module docstring
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.basic.neotraits.api import HasTraits, Attr, NArray

TRACTS_CHUNK_SIZE = 100


class Tracts(HasTraits):
    """Datatype for results of diffusion imaging tractography."""

    vertices = NArray(
        label="Vertex positions",
        doc="""An array specifying coordinates for the tracts vertices."""
    )

    tract_start_idx = NArray(
        dtype=int,
        label="Tract starting indices",
        doc="""Where is the first vertex of a tract in the vertex array"""
    )

    tract_region = NArray(
        dtype=int,
        label="Tract region index",
        required=False,
        doc="""
            An index used to find quickly all tract emerging from a region
            tract_region[i] is the region of the i'th tract. -1 represents the background
            """
    )

    region_volume_map = Attr(
        field_type=RegionVolumeMapping,
        label="Region volume Mapping used to create the tract_region index",
    )

    @property
    def tracts_count(self):
        return len(self.tract_start_idx) - 1
