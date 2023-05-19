# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
TVB generic configurations are here.

E.g. Scientific libraries modules are plugged, to avoid close dependencies.
E.g. A list with all the modules where adapters are implemented.
"""
# There are kept here for being used inside tvb.core
# We can not move all these in IntrospectovRegistry, due to circular dependencies

from tvb.analyzers import METRICS

ALGORITHMS = METRICS

SIMULATION_DATATYPE_CLASS = "SimulationState"

TVB_IMPORTER_MODULE = "tvb.adapters.uploaders.tvb_importer"
TVB_IMPORTER_CLASS = "TVBImporter"

SIMULATOR_MODULE = "tvb.adapters.simulator.simulator_adapter"
SIMULATOR_CLASS = "SimulatorAdapter"

CONNECTIVITY_CREATOR_MODULE = 'tvb.adapters.creators.connectivity_creator'
CONNECTIVITY_CREATOR_CLASS = 'ConnectivityCreator'

MEASURE_METRICS_MODULE = 'tvb.adapters.analyzers.metrics_group_timeseries'
MEASURE_METRICS_CLASS = 'TimeseriesMetricsAdapter'
MEASURE_METRICS_MODEL_CLASS = 'TimeseriesMetricsAdapterModel'

DEFAULT_PROJECT_GID = '2cc58a73-25c1-11e5-a7af-14109fe3bf71'

VIEW_MODEL2ADAPTER = {}

DATATYPE_MEASURE_INDEX_MODULE = 'tvb.adapters.datatypes.db.mapped_value'
DATATYPE_MEASURE_INDEX_CLASS = 'DatatypeMeasureIndex'
