# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

ADAPTERS = {"Analyze": {'modules': ["tvb.adapters.analyzers"],
                        'launchable': True, 'defaultdatastate': 'INTERMEDIATE', 'order_nr': '1'},
            "Simulate": {'modules': ["tvb.adapters.simulator"],
                         'defaultdatastate': 'INTERMEDIATE', 'order_nr': '0'},
            "Upload": {'modules': ["tvb.adapters.uploaders"],
                       'rawinput': True, 'defaultdatastate': 'RAW_DATA'},
            "View": {'modules': ["tvb.adapters.visualizers"],
                     'launchable': True, 'display': True, 'defaultdatastate': 'INTERMEDIATE', 'order_nr': '3'},
            "Create": {'modules': ["tvb.adapters.creators"],
                       'defaultdatastate': 'RAW_DATA', 'order_nr': '0'}
            }

DATATYPES_PATH = ["tvb.datatypes"]
REMOVERS_PATH = ["tvb.datatype_removers"]
PORTLETS_PATH = ["tvb.adapters.portlets"]

SIMULATOR_MODULE = "tvb.adapters.simulator.simulator_adapter"
SIMULATOR_CLASS = "SimulatorAdapter"

SIMULATION_DATATYPE_MODULE = 'tvb.datatypes.simulation_state'
SIMULATION_DATATYPE_CLASS = "SimulationState"

CONNECTIVITY_MODULE = 'tvb.adapters.visualizers.connectivity'
CONNECTIVITY_CLASS = 'ConnectivityViewer'

ALLEN_CREATOR_MODULE = 'tvb.adapters.creators.allen_creator'
ALLEN_CREATOR_CLASS = 'AllenConnectomeBuilder'

MEASURE_METRICS_MODULE = "tvb.adapters.analyzers.metrics_group_timeseries"
MEASURE_METRICS_CLASS = "TimeseriesMetricsAdapter"

DISCRETE_PSE_ADAPTER_MODULE = "tvb.adapters.visualizers.pse_discrete"
DISCRETE_PSE_ADAPTER_CLASS = "DiscretePSEAdapter"

ISOCLINE_PSE_ADAPTER_MODULE = "tvb.adapters.visualizers.pse_isocline"
ISOCLINE_PSE_ADAPTER_CLASS = "IsoclinePSEAdapter"

TVB_IMPORTER_MODULE = "tvb.adapters.uploaders.tvb_importer"
TVB_IMPORTER_CLASS = "TVBImporter"

CONNECTIVITY_CREATOR_MODULE = 'tvb.adapters.creators.connectivity_creator'
CONNECTIVITY_CREATOR_CLASS = 'ConnectivityCreator'

#DEFAULT_PORTLETS = {$tab_index: {$select_index_in_tab: '$portlet_identifier'}}
DEFAULT_PORTLETS = {0: {0: 'TimeSeries'}}

DEFAULT_PROJECT_GID = '2cc58a73-25c1-11e5-a7af-14109fe3bf71'
