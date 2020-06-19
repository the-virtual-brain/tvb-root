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
TVB generic configurations are here.

E.g. Scientific libraries modules are plugged, to avoid close dependencies.
E.g. A list with all the modules where adapters are implemented.
"""
# There are kept here for being used inside tvb.core
# We can not move all these in IntrospectovRegistry, due to circular dependencies

from collections import OrderedDict
# Import metrics here, so that Traits will find them and return them as known subclasses
import tvb.analyzers.metric_kuramoto_index
import tvb.analyzers.metric_proxy_metastability
import tvb.analyzers.metric_variance_global
import tvb.analyzers.metric_variance_of_node_variance
from tvb.analyzers.metrics_base import BaseTimeseriesMetricAlgorithm

ALGORITHMS = BaseTimeseriesMetricAlgorithm.get_known_subclasses(include_itself=False)

algo_names = list(ALGORITHMS)
algo_names.sort()
choices = OrderedDict()
for name in algo_names:
    choices[name] = name

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

DEFAULT_PORTLETS = {0: {0: 'TimeSeries'}}