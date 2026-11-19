# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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


class HybridSimulatorURLs(object):
    SET_CONNECTIVITY_URL = '/burst/hybrid/set_connectivity'
    # the wizard step listing the saved Subnetworks, shown in the cockpit configuration column
    SET_SUBNETWORKS_URL = '/burst/hybrid/set_subnetworks'
    # the board on which the regions are grouped, shown in the contextual configuration column
    CONFIGURE_SUBNETWORKS_URL = '/burst/hybrid/configure_subnetworks'
    SAVE_SUBNETWORKS_URL = '/burst/hybrid/save_subnetworks'
    ADD_SUBNETWORK_URL = '/burst/hybrid/add_subnetwork'
    REMOVE_SUBNETWORK_URL = '/burst/hybrid/remove_subnetwork'
    RENAME_SUBNETWORK_URL = '/burst/hybrid/rename_subnetwork'
    MOVE_REGIONS_URL = '/burst/hybrid/move_regions'
    SET_FRAGMENT_URL = '/burst/hybrid/set_fragment_url'
    # what every url of the Hybrid Simulator starts with, the steps remembered for a page reload included
    HYBRID_URL_PREFIX = '/burst/hybrid/'
    # the wizard step under which every Subnetwork's dynamics are configured, holding the shared dt
    SET_SUBNETWORK_DYNAMICS_URL = '/burst/hybrid/set_subnetwork_dynamics'
    SELECT_SUBNETWORK_URL = '/burst/hybrid/select_subnetwork'
    # the editor of the selected Subnetwork's dynamics, re-rendered from the draft after a class change
    SUBNETWORK_EDITOR_URL = '/burst/hybrid/subnetwork_editor'
    SAVE_SUBNETWORK_DYNAMICS_URL = '/burst/hybrid/save_subnetwork_dynamics'
    # the sections of the Subnetwork editor, the same forms as the classic Cockpit chain. Each posts to
    # its own url, which applies it onto the draft of the selected Subnetwork.
    SET_SUBNETWORK_MODEL_URL = '/burst/hybrid/set_subnetwork_model'
    SET_SUBNETWORK_MODEL_PARAMS_URL = '/burst/hybrid/set_subnetwork_model_params'
    SET_SUBNETWORK_INTEGRATOR_URL = '/burst/hybrid/set_subnetwork_integrator'
    SET_SUBNETWORK_INTEGRATOR_PARAMS_URL = '/burst/hybrid/set_subnetwork_integrator_params'
    SET_SUBNETWORK_NOISE_PARAMS_URL = '/burst/hybrid/set_subnetwork_noise_params'
    SET_SUBNETWORK_NOISE_EQUATION_PARAMS_URL = '/burst/hybrid/set_subnetwork_noise_equation_params'
    # placing saved Dynamics on the regions of the Subnetwork being configured, in the third column
    CONFIGURE_REGION_MODEL_URL = '/burst/hybrid/configure_region_model'
    APPLY_REGION_MODEL_URL = '/burst/hybrid/apply_region_model'
    SUBMIT_REGION_MODEL_URL = '/burst/hybrid/submit_region_model'
    # the wizard step generating the Projections out of the Connectivity and the Subnetwork grouping
    SET_PROJECTIONS_URL = '/burst/hybrid/set_projections'
    # the Projections editor, in the third column: choosing a Projection, editing it, leaving it out
    CONFIGURE_PROJECTIONS_URL = '/burst/hybrid/configure_projections'
    SET_PROJECTION_URL = '/burst/hybrid/set_projection'
    SELECT_PROJECTION_URL = '/burst/hybrid/select_projection'
    SELECT_INTER_PROJECTION_URL = '/burst/hybrid/select_inter_projection'
    SET_PROJECTION_INCLUDED_URL = '/burst/hybrid/set_projection_included'
    # the global configuration: what the simulation records, for how long, and what that produces
    SET_MONITORS_URL = '/burst/hybrid/set_monitors'
    # opens the Monitors step without submitting it, which is what moving on from the Projections does
    OPEN_MONITORS_URL = '/burst/hybrid/open_monitors'
    SET_MONITOR_PARAMS_URL = '/burst/hybrid/set_monitor_params'
    SET_MONITOR_EQUATION_URL = '/burst/hybrid/set_monitor_equation'
    SET_SIMULATION_SUMMARY_URL = '/burst/hybrid/set_simulation_summary'
