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

EVENTS_FOLDER = "tvb.config"

PORTLETS_PATH = ["tvb.adapters.portlets"]

SIMULATOR_MODULE = "tvb.adapters.simulator.simulator_adapter"
SIMULATOR_CLASS = "SimulatorAdapter"

SIMULATION_DATATYPE_MODULE = 'tvb.datatypes.simulation_state'
SIMULATION_DATATYPE_CLASS = "SimulationState"

CONNECTIVITY_MODULE = 'tvb.adapters.visualizers.connectivity'
CONNECTIVITY_CLASS = 'ConnectivityViewer'

MEASURE_METRICS_MODULE = "tvb.adapters.analyzers.metrics_group_timeseries"
MEASURE_METRICS_CLASS = "TimeseriesMetricsAdapter"

DISCRETE_PSE_ADAPTER_MODULE = "tvb.adapters.visualizers.pse_discrete"
DISCRETE_PSE_ADAPTER_CLASS = "DiscretePSEAdapter"

ISOCLINE_PSE_ADAPTER_MODULE = "tvb.adapters.visualizers.pse_isocline"
ISOCLINE_PSE_ADAPTER_CLASS = "IsoclinePSEAdapter"

TVB_IMPORTER_MODULE = "tvb.adapters.uploaders.tvb_importer"
TVB_IMPORTER_CLASS = "TVBImporter"

#DEFAULT_PORTLETS = {$tab_index: {$select_index_in_tab: '$portlet_identifier'}}
DEFAULT_PORTLETS = {0: {0: 'TimeSeries'}}


