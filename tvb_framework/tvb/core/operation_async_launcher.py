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
This module is called in a new process by the rpserver:
Example: python operation_async_launcher.py 4 user_name_label
4 is the operation id stored in the DataBase in the table "OPERATIONS"
It gets the algorithm, and the adapter with its parameters from database.
And finally launches the computation.
The results of the computation will be stored by the adapter itself.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@tvb.invalid>

"""

import sys
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_operation import has_finished
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.services.operation_service import OperationService
from tvb.core.services.burst_service import BurstService
from tvb.storage.storage_interface import StorageInterface

if __name__ == '__main__':
    TvbProfile.set_profile(sys.argv[2], True)


def do_operation_launch(operation_id):
    """
    Event attached to the local queue for executing an operation, when we will have resources available.
    """
    log = get_logger('tvb.core.operation_async_launcher')
    burst_service = BurstService()

    try:
        log.debug("Loading operation with id=%s" % operation_id)
        curent_operation = dao.get_operation_by_id(operation_id)
        stored_adapter = curent_operation.algorithm
        log.debug("Importing Algorithm: " + str(stored_adapter.classname) +
                  " for Operation:" + str(curent_operation.id))
        adapter_instance = ABCAdapter.build_adapter(stored_adapter)
        # Un-comment bellow for profiling an operation:
        # import cherrypy.lib.profiler as profiler
        # p = profiler.Profiler("/Users/lia.domide/TVB/profiler/")
        # p.run(OperationService().initiate_prelaunch, curent_operation, adapter_instance, {}, **PARAMS)

        OperationService().initiate_prelaunch(curent_operation, adapter_instance)
        if curent_operation.fk_operation_group:
            parent_burst = dao.get_generic_entity(BurstConfiguration, curent_operation.fk_operation_group,
                                                  'fk_operation_group')[0]
            operations_in_group = dao.get_operations_in_group(curent_operation.fk_operation_group)
            if parent_burst.fk_metric_operation_group:
                operations_in_group.extend(dao.get_operations_in_group(parent_burst.fk_metric_operation_group))
            burst_finished = True
            for operation in operations_in_group:
                if not has_finished(operation.status):
                    burst_finished = False
                    break

            if burst_finished and parent_burst is not None and parent_burst.status != BurstConfiguration.BURST_ERROR:
                burst_service.mark_burst_finished(parent_burst)
        else:
            parent_burst = burst_service.get_burst_for_operation_id(operation_id)
            if parent_burst is not None:
                burst_service.mark_burst_finished(parent_burst)

        log.debug("Successfully finished operation " + str(operation_id))

    except Exception as excep:
        log.error("Could not execute operation " + str(operation_id))
        log.exception(excep)
        parent_burst = burst_service.get_burst_for_operation_id(operation_id)
        if parent_burst is not None:
            burst_service.mark_burst_finished(parent_burst, error_message=str(excep))


if __name__ == '__main__':
    OPERATION_ID = sys.argv[1]
    storage_interface = StorageInterface()
    if storage_interface.app_encryption_handler():
        storage_interface.start()

    do_operation_launch(OPERATION_ID)

    if storage_interface.app_encryption_handler():
        storage_interface.mark_stop()
        storage_interface.join()
