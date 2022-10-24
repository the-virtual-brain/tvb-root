# coding=utf-8
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

from tvb.config.init.datatypes_registry import REGISTRY, populate_datatypes_registry
from tvb.adapters.datatypes.h5.time_series_h5 import *
from tvb.adapters.datatypes.db.time_series import *

from tvb.contrib.scripts.datatypes.time_series import *


populate_datatypes_registry()


REGISTRY.register_datatype(TimeSeries, TimeSeriesH5, TimeSeriesIndex)
REGISTRY.register_datatype(TimeSeriesRegion, TimeSeriesRegionH5, TimeSeriesRegionIndex)
REGISTRY.register_datatype(TimeSeriesSurface, TimeSeriesSurfaceH5, TimeSeriesSurfaceIndex)
REGISTRY.register_datatype(TimeSeriesVolume, TimeSeriesVolumeH5, TimeSeriesVolumeIndex)
REGISTRY.register_datatype(TimeSeriesEEG, TimeSeriesEEGH5, TimeSeriesEEGIndex)
REGISTRY.register_datatype(TimeSeriesMEG, TimeSeriesMEGH5, TimeSeriesMEGIndex)
REGISTRY.register_datatype(TimeSeriesSEEG, TimeSeriesSEEGH5, TimeSeriesSEEGIndex)
