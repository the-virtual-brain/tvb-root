# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
This is the module where all TVB Index DataTypes are hooked into the framework.

Define in __all__ attribute, modules to be introspected for finding their classes.

"""
from tvb.adapters.datatypes.db.removers.remover_connectivity import ConnectivityRemover
from tvb.adapters.datatypes.db.removers.remover_region_mapping import RegionMappingRemover, RegionVolumeMappingRemover
from tvb.adapters.datatypes.db.removers.remover_sensor import SensorRemover
from tvb.adapters.datatypes.db.removers.remover_surface import SurfaceRemover
from tvb.adapters.datatypes.db.removers.remover_timeseries import TimeseriesRemover
from tvb.adapters.datatypes.db.removers.remover_volume import VolumeRemover

ALL_DATATYPES = ["annotation", "connectivity", "fcd", "graph", "local_connectivity", "mapped_value",
                 "mode_decompositions", "patterns", "projections", "region_mapping", "sensors", "spectral",
                 "surface", "temporal_correlations", "time_series", "tracts", "volume"]

DATATYPE_REMOVERS = {
    "ConnectivityIndex": ConnectivityRemover,
    "SurfaceIndex": SurfaceRemover,
    "SensorsIndex": SensorRemover,
    "VolumeIndex": VolumeRemover,
    "RegionMappingIndex": RegionMappingRemover,
    "RegionVolumeMappingIndex": RegionVolumeMappingRemover,
    "TimeSeriesIndex": TimeseriesRemover
}
