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
Load a simulation from JSON file (possibly to export from web GUI) and launch it (together with analyzers, eventually).
A "burst" on the simulation page history should appear after running this.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)


import json
from tvb.basic.logger.builder import get_logger
from tvb.config import SIMULATOR_MODULE, SIMULATOR_CLASS
from tvb.core.entities.storage import dao
from tvb.core.services.flow_service import FlowService
from tvb.core.services.import_service import ImportService
from tvb.core.services.burst_service import BurstService, LAUNCH_NEW

LOG = get_logger(__name__)


def launch_simulation_workflow(json_path, prj_id):
    """

    :param json_path: Path towards a local JSON file exported from GUI
    :param prj_id: This ID of a project needs to exists in DB, and it can be taken from the WebInterface
    """
    project = dao.get_project_by_id(prj_id)

    with open(json_path, 'rb') as input_file:
        simulation_json = input_file.read()
        simulation_json = json.loads(simulation_json)
        LOG.info("Simulation JSON loaded from file '%s': \n  %s", json_path, simulation_json)

        importer = ImportService()
        simulation_config = importer.load_burst_entity(simulation_json, prj_id)
        LOG.info("Simulation Workflow configuration object loaded: \n  %s", simulation_config)

        flow_service = FlowService()
        stored_adapter = flow_service.get_algorithm_by_module_and_class(SIMULATOR_MODULE, SIMULATOR_CLASS)
        LOG.info("Found Simulation algorithm in local DB: \n   %s", stored_adapter)

        burst_service = BurstService()
        burst_service.launch_burst(simulation_config, 0, stored_adapter.id, project.administrator.id, LAUNCH_NEW)
        LOG.info("Check in the web GUI for your operation. It should be starting now ...")


## Before starting this, we need to have TVB web interface launched at least once (to have a default project, user, etc)
if __name__ == "__main__":
    launch_simulation_workflow("tvb_simulation_3.json", 1)
