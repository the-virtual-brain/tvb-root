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

from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
from tvb.interfaces.web.controllers import common


class HybridSimulatorContext(object):
    KEY_HYBRID_SIMULATOR_CONFIG = 'hybrid_simulator_configuration'
    KEY_LAST_LOADED_FORM_URL = 'hybrid_simulator_last_loaded_form_url'
    # The grouping being edited in the contextual configuration column. It is kept apart from the
    # Hybrid Simulator configuration so that the wizard keeps showing the last saved grouping until
    # the user explicitly saves the one on the board.
    KEY_SUBNETWORKS_DRAFT = 'hybrid_simulator_subnetworks_draft'
    # The Model/Integrator configuration being edited, keyed by Subnetwork identifier. Held apart from
    # the Hybrid Simulator configuration for the same reason the grouping draft is: the wizard step keeps
    # summarising the saved dynamics until the user explicitly saves the ones being edited.
    KEY_DYNAMICS_DRAFT = 'hybrid_simulator_dynamics_draft'
    # Identifier of the Subnetwork whose dynamics the contextual column is currently configuring
    KEY_SELECTED_SUBNETWORK = 'hybrid_simulator_selected_subnetwork'
    # Which saved Dynamic is placed on which region, per Subnetwork, for the Region Model panel. Kept
    # next to the dynamics draft rather than derived from the Model, since the parameter arrays alone
    # cannot say which Dynamic produced them.
    KEY_REGION_MODEL = 'hybrid_simulator_region_model'

    @property
    def project(self):
        return common.get_current_project()

    @property
    def logged_user(self):
        return common.get_logged_user()

    @property
    def hybrid_simulator(self):
        return common.get_from_session(self.KEY_HYBRID_SIMULATOR_CONFIG)

    @property
    def last_loaded_fragment_url(self):
        return common.get_from_session(self.KEY_LAST_LOADED_FORM_URL)

    @property
    def subnetworks_draft(self):
        return common.get_from_session(self.KEY_SUBNETWORKS_DRAFT)

    @property
    def dynamics_draft(self):
        return common.get_from_session(self.KEY_DYNAMICS_DRAFT)

    @property
    def selected_subnetwork(self):
        return common.get_from_session(self.KEY_SELECTED_SUBNETWORK)

    @property
    def region_model(self):
        return common.get_from_session(self.KEY_REGION_MODEL)

    def set_hybrid_simulator(self, hybrid_simulator=None):
        if not hybrid_simulator and not self.hybrid_simulator:
            hybrid_simulator = HybridSimulatorAdapterModel()
        if hybrid_simulator:
            common.add2session(self.KEY_HYBRID_SIMULATOR_CONFIG, hybrid_simulator)

    @staticmethod
    def set_subnetworks_draft(subnetworks):
        common.add2session(HybridSimulatorContext.KEY_SUBNETWORKS_DRAFT, subnetworks)

    @staticmethod
    def clear_subnetworks_draft():
        common.remove_from_session(HybridSimulatorContext.KEY_SUBNETWORKS_DRAFT)

    @staticmethod
    def set_dynamics_draft(dynamics_by_id):
        common.add2session(HybridSimulatorContext.KEY_DYNAMICS_DRAFT, dynamics_by_id)

    @staticmethod
    def set_region_model(region_model):
        common.add2session(HybridSimulatorContext.KEY_REGION_MODEL, region_model)

    @staticmethod
    def clear_dynamics_draft():
        common.remove_from_session(HybridSimulatorContext.KEY_DYNAMICS_DRAFT)
        common.remove_from_session(HybridSimulatorContext.KEY_SELECTED_SUBNETWORK)
        common.remove_from_session(HybridSimulatorContext.KEY_REGION_MODEL)

    @staticmethod
    def set_selected_subnetwork(subnetwork_id):
        common.add2session(HybridSimulatorContext.KEY_SELECTED_SUBNETWORK, subnetwork_id)

    def reset_hybrid_simulator(self):
        self.set_hybrid_simulator(HybridSimulatorAdapterModel())
        self.clear_subnetworks_draft()
        self.clear_dynamics_draft()

    @staticmethod
    def add_last_loaded_form_url_to_session(last_loaded_form_url):
        common.add2session(HybridSimulatorContext.KEY_LAST_LOADED_FORM_URL, last_loaded_form_url)

    @staticmethod
    def clean_project_data_from_session():
        common.remove_from_session(HybridSimulatorContext.KEY_HYBRID_SIMULATOR_CONFIG)
        common.remove_from_session(HybridSimulatorContext.KEY_LAST_LOADED_FORM_URL)
        common.remove_from_session(HybridSimulatorContext.KEY_SUBNETWORKS_DRAFT)
        common.remove_from_session(HybridSimulatorContext.KEY_DYNAMICS_DRAFT)
        common.remove_from_session(HybridSimulatorContext.KEY_SELECTED_SUBNETWORK)
        common.remove_from_session(HybridSimulatorContext.KEY_REGION_MODEL)
