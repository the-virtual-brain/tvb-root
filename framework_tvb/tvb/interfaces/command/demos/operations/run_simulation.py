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
Find a Connectivity in current project (by Subject) and later run a simulation on it.

__main__ will contain the code.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

import sys
from time import sleep
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.flow_service import FlowService
from tvb.core.services.operation_service import OperationService
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.datatypes.connectivity import Connectivity


LOG = get_logger(__name__)


## Before starting this, we need to have TVB web interface launched at least once (to have a default project, user, etc)
if __name__ == "__main__":

    flow_service = FlowService()
    operation_service = OperationService()

    ## This ID of a project needs to exists in DB, and it can be taken from the WebInterface:
    project = dao.get_project_by_id(1)

    ## Prepare the Adapter
    adapter_instance = ABCAdapter.build_adapter_from_class(SimulatorAdapter)

    ## Prepare the input algorithms as if they were coming from web UI submit:
    ## TODO create helper methods for working with objects instead of strings.
    connectivity = dao.get_generic_entity(Connectivity, DataTypeMetaData.DEFAULT_SUBJECT, "subject")[0]
    launch_args = dict()
    for f in adapter_instance.flaten_input_interface():
        launch_args[f["name"]] = str(f["default"]) if 'default' in f else None
    launch_args["connectivity"] = connectivity.gid
    launch_args["model_parameters_option_Generic2dOscillator_variables_of_interest"] = 'V'

    if len(sys.argv) > 1:
        launch_args["model_parameters_option_Generic2dOscillator_tau"] = sys.argv[1]

    ## launch an operation and have the results stored both in DB and on disk
    launched_operation = flow_service.fire_operation(adapter_instance, project.administrator,
                                                     project.id, **launch_args)[0]

    ## wait for the operation to finish
    while not launched_operation.has_finished:
        sleep(5)
        launched_operation = dao.get_operation_by_id(launched_operation.id)

    if launched_operation.status == model.STATUS_FINISHED:
        ts = dao.get_generic_entity(TimeSeriesRegion, launched_operation.id, "fk_from_operation")[0]
        LOG.info("TimeSeries result is: %s " % ts.summary_info)
        print(ts.summary_info)
    else:
        LOG.warning("Operation ended with problems [%s]: [%s]" % (launched_operation.status,
                                                                  launched_operation.additional_info))


