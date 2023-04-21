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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@invalid.tvb>
"""

import os
from subprocess import Popen, PIPE
from threading import Thread

from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model.model_operation import OperationProcessIdentifier, STATUS_CANCELED
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.backend_clients.backend_client import BackendClient
from tvb.core.services.burst_service import BurstService
from tvb.core.utils import parse_json_parameters

LOGGER = get_logger(__name__)


class ClusterSchedulerClient(BackendClient):
    # TODO: fix this with neoforms
    """
    Simple class, to mimic the same behavior we are expecting from StandAloneClient, but firing behind
    the cluster job scheduling process..
    """

    @staticmethod
    def _run_cluster_job(operation_identifier, user_name_label, adapter_instance):
        """
        Threaded Popen
        It is the function called by the ClusterSchedulerClient in a Thread.
        This function starts a new process.
        """
        # Load operation so we can estimate the execution time
        operation = dao.get_operation_by_id(operation_identifier)
        view_model = h5.load_view_model(operation)
        # kwargs = adapter_instance.prepare_ui_inputs(kwargs)
        time_estimate = int(adapter_instance.get_execution_time_approximation(view_model))
        hours = int(time_estimate / 3600)
        minutes = (int(time_estimate) % 3600) / 60
        seconds = int(time_estimate) % 60
        # Anything lower than 5 hours just use default walltime
        if hours < 5:
            walltime = "05:00:00"
        else:
            if hours < 10:
                hours = "0%d" % hours
            else:
                hours = str(hours)
            walltime = "%s:%s:%s" % (hours, str(minutes), str(seconds))

        call_arg = TvbProfile.current.cluster.SCHEDULE_COMMAND % (operation_identifier, user_name_label, walltime)
        LOGGER.info(call_arg)
        process_ = Popen([call_arg], stdout=PIPE, shell=True)
        job_id = process_.stdout.read().replace('\n', '').split(TvbProfile.current.cluster.JOB_ID_STRING)[-1]
        LOGGER.info("Got jobIdentifier = %s for CLUSTER operationID = %s" % (job_id, operation_identifier))
        operation_identifier = OperationProcessIdentifier(operation_identifier, job_id=job_id)
        dao.store_entity(operation_identifier)

    @staticmethod
    def execute(operation_id, user_name_label, adapter_instance):
        """Call the correct system command to submit a job to the cluster."""
        thread = Thread(target=ClusterSchedulerClient._run_cluster_job,
                        kwargs={'operation_identifier': operation_id,
                                'user_name_label': user_name_label,
                                'adapter_instance': adapter_instance})
        thread.start()

    @staticmethod
    def stop_operation(operation_id):
        """
        Stop a thread for a given operation id
        """
        operation = dao.try_get_operation_by_id(operation_id)
        if not operation or operation.has_finished:
            LOGGER.warning("Operation already stopped or not found is given to stop job: %s" % operation_id)
            return True

        operation_process = dao.get_operation_process_for_operation(operation_id)
        result = 0
        # Try to kill only if operation job process is not None
        if operation_process is not None:
            stop_command = TvbProfile.current.cluster.STOP_COMMAND % operation_process.job_id
            LOGGER.info("Stopping cluster operation: %s" % stop_command)
            result = os.system(stop_command)
            if result != 0:
                LOGGER.error("Stopping cluster operation was unsuccessful. Try following status with '" +
                             TvbProfile.current.cluster.STATUS_COMMAND + "'" % operation_process.job_id)

        BurstService().persist_operation_state(operation, STATUS_CANCELED)

        return result == 0
