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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@invalid.tvb>
"""
import json
import os
import sys
import signal
import queue as queue
import threading
from subprocess import Popen, PIPE
from time import sleep
import pyunicore.client as unicore_client
from tvb.adapters.simulator.hpc_simulator_adapter import HPCSimulatorAdapter
from tvb.basic.config.settings import HPCSettings
from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.simulator_h5 import SimulatorH5
from tvb.core.neocom import h5
from tvb.core.neotraits.h5 import H5File
from tvb.core.services.burst_service import BurstService
from tvb.core.utils import parse_json_parameters
from tvb.core.entities.model.model_operation import OperationProcessIdentifier, STATUS_ERROR, STATUS_CANCELED
from tvb.core.entities.storage import dao

LOGGER = get_logger(__name__)

CURRENT_ACTIVE_THREADS = []

LOCKS_QUEUE = queue.Queue(0)
for i in range(TvbProfile.current.MAX_THREADS_NUMBER):
    LOCKS_QUEUE.put(1)


class OperationExecutor(threading.Thread):
    """
    Thread in charge for starting an operation, used both on cluster and with stand-alone installations.
    """

    def __init__(self, op_id):
        threading.Thread.__init__(self)
        self.operation_id = op_id
        self._stop_ev = threading.Event()

    def run(self):
        """
        Get the required data from the operation queue and launch the operation.
        """
        # Try to get a spot to launch own operation.
        LOCKS_QUEUE.get(True)
        operation_id = self.operation_id
        run_params = [TvbProfile.current.PYTHON_INTERPRETER_PATH, '-m', 'tvb.core.operation_async_launcher',
                      str(operation_id), TvbProfile.CURRENT_PROFILE_NAME]

        # In the exceptional case where the user pressed stop while the Thread startup is done,
        # We should no longer launch the operation.
        if self.stopped() is False:

            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(sys.path)
            # anything that was already in $PYTHONPATH should have been reproduced in sys.path

            launched_process = Popen(run_params, stdout=PIPE, stderr=PIPE, env=env)

            LOGGER.debug("Storing pid=%s for operation id=%s launched on local machine." % (operation_id,
                                                                                            launched_process.pid))
            op_ident = OperationProcessIdentifier(operation_id, pid=launched_process.pid)
            dao.store_entity(op_ident)

            if self.stopped():
                # In the exceptional case where the user pressed stop while the Thread startup is done.
                # and stop_operation is concurrently asking about OperationProcessIdentity.
                self.stop_pid(launched_process.pid)

            subprocess_result = launched_process.communicate()
            LOGGER.info("Finished with launch of operation %s" % operation_id)
            returned = launched_process.wait()

            if returned != 0 and not self.stopped():
                # Process did not end as expected. (e.g. Segmentation fault)
                burst_service = BurstService()
                operation = dao.get_operation_by_id(self.operation_id)
                LOGGER.error("Operation suffered fatal failure! Exit code: %s Exit message: %s" % (returned,
                                                                                                   subprocess_result))

                burst_service.persist_operation_state(operation, STATUS_ERROR,
                                                      "Operation failed unexpectedly! Please check the log files.")

                burst_entity = dao.get_burst_for_operation_id(self.operation_id)
                if burst_entity:
                    message = "Error in operation process! Possibly segmentation fault."
                    burst_service.mark_burst_finished(burst_entity, error_message=message)

            del launched_process

        # Give back empty spot now that you finished your operation
        CURRENT_ACTIVE_THREADS.remove(self)
        LOCKS_QUEUE.put(1)

    def _stop(self):
        """ Mark current thread for stop"""
        self._stop_ev.set()

    def stopped(self):
        """Check if current thread was marked for stop."""
        return self._stop_ev.isSet()

    @staticmethod
    def stop_pid(pid):
        """
        Stop a process specified by PID.
        :returns: True when specified Process was stopped in here, \
                  False in case of exception(e.g. process stopped in advance).
        """
        if sys.platform == 'win32':
            try:
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(1, False, int(pid))
                ctypes.windll.kernel32.TerminateProcess(handle, -1)
                ctypes.windll.kernel32.CloseHandle(handle)
            except OSError:
                return False
        else:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                return False

        return True


class StandAloneClient(object):
    """
    Instead of communicating with a back-end cluster, fire locally a new thread.
    """

    @staticmethod
    def execute(operation_id, user_name_label, adapter_instance):
        """Start asynchronous operation locally"""
        thread = OperationExecutor(operation_id)
        CURRENT_ACTIVE_THREADS.append(thread)
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

        LOGGER.debug("Stopping operation: %s" % str(operation_id))

        # Set the thread stop flag to true
        for thread in CURRENT_ACTIVE_THREADS:
            if int(thread.operation_id) == operation_id:
                thread._stop()
                LOGGER.debug("Found running thread for operation: %d" % operation_id)

        # Kill Thread
        stopped = True
        operation_process = dao.get_operation_process_for_operation(operation_id)
        if operation_process is not None:
            # Now try to kill the operation if it exists
            stopped = OperationExecutor.stop_pid(operation_process.pid)
            if not stopped:
                LOGGER.debug("Operation %d was probably killed from it's specific thread." % operation_id)
            else:
                LOGGER.debug("Stopped OperationExecutor process for %d" % operation_id)

        # Mark operation as canceled in DB and on disk
        BurstService().persist_operation_state(operation, STATUS_CANCELED)

        return stopped


class ClusterSchedulerClient(object):
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
        kwargs = parse_json_parameters(operation.parameters)
        # kwargs = adapter_instance.prepare_ui_inputs(kwargs)
        time_estimate = int(adapter_instance.get_execution_time_approximation(**kwargs))
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
        thread = threading.Thread(target=ClusterSchedulerClient._run_cluster_job,
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


class HPCSchedulerClient(object):
    """
    Simple class, to mimic the same behavior we are expecting from StandAloneClient, but firing the operation on
    an HPC node. Define TVB_BIN_ENV_KEY and CSCS_LOGIN_TOKEN_ENV_KEY as environment variables before running on HPC.
    """
    TVB_BIN_ENV_KEY = 'TVB_BIN'
    CSCS_LOGIN_TOKEN_ENV_KEY = 'CSCS_LOGIN_TOKEN'

    @staticmethod
    def _gather_file_list(h5_file, file_list):
        references = h5_file.gather_references_gids()
        for reference in references:
            if reference is None:
                continue
            try:
                # TODO: nicer way to identify files?
                index = dao.get_datatype_by_gid(reference.hex)
                reference_h5_file = h5.path_for_stored_index(index)
                h5_file_class = h5.h5_file_for_index(index)
            except Exception:
                reference_h5_file = h5_file.get_reference_path(reference)
                h5_file_class = H5File.from_file(reference_h5_file)
            if reference_h5_file not in file_list:
                file_list.append(reference_h5_file)
            HPCSchedulerClient._gather_file_list(h5_file_class, file_list)
            h5_file_class.close()

    @staticmethod
    def _prepare_input(operation, input_gid):
        storage_path = FilesHelper().get_project_folder(operation.project, str(operation.id))
        simulator_in_path = h5.path_for(storage_path, SimulatorH5, input_gid)

        input_files_list = []
        with SimulatorH5(simulator_in_path) as simulator_h5:
            HPCSchedulerClient._gather_file_list(simulator_h5, input_files_list)
        input_files_list.append(simulator_in_path)
        return input_files_list

    @staticmethod
    def _configure_job(simulator_gid, available_space):
        bash_entrypoint = os.path.join(os.environ[HPCSchedulerClient.TVB_BIN_ENV_KEY],
                                       HPCSettings.HPC_LAUNCHER_SH_SCRIPT)
        job_inputs = [bash_entrypoint]

        # Build job configuration JSON
        my_job = {}
        my_job[HPCSettings.UNICORE_EXE_KEY] = os.path.basename(bash_entrypoint)
        my_job[HPCSettings.UNICORE_ARGS_KEY] = [simulator_gid, available_space]
        my_job[HPCSettings.UNICORE_RESOURCER_KEY] = {"CPUs": "1"}

        return my_job, job_inputs

    @staticmethod
    def _run_hpc_job(operation_identifier):
        operation = dao.get_operation_by_id(operation_identifier)
        input_gid = json.loads(operation.parameters)['gid']

        job_inputs = HPCSchedulerClient._prepare_input(operation, input_gid)

        transport = unicore_client.Transport(os.environ[HPCSchedulerClient.CSCS_LOGIN_TOKEN_ENV_KEY])
        registry = unicore_client.Registry(transport, unicore_client._HBP_REGISTRY_URL)

        # use "DAINT-CSCS" -- change if another supercomputer is prepared for usage
        site_client = registry.site(HPCSettings.SUPERCOMPUTER_SITE)

        # TODO: compute available space
        job_config, job_extra_inputs = HPCSchedulerClient._configure_job(input_gid, '100')
        job_inputs.extend(job_extra_inputs)
        job = site_client.new_job(job_description=job_config, inputs=job_inputs)
        LOGGER.info(job.properties)
        while job.is_running():
            LOGGER.info(job.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY])
            sleep(10)
        LOGGER.info(job.properties[HPCSettings.JOB_STATUS_KEY])
        wd = job.working_dir
        output_list = wd.listdir(HPCSimulatorAdapter.OUTPUT_FOLDER)
        for output_filename, output_filepath in output_list.items():
            output_filepath.download(output_filename)
        LOGGER.info(wd.properties)
        LOGGER.info(wd.listdir())

    @staticmethod
    def execute(operation_id, user_name_label, adapter_instance):
        """Call the correct system command to submit a job to HPC."""
        thread = threading.Thread(target=HPCSchedulerClient._run_hpc_job,
                                  kwargs={'operation_identifier': operation_id})
        thread.start()

    @staticmethod
    def stop_operation(operation_id):
        pass


if TvbProfile.current.hpc.IS_HPC_RUN:
    # Return an entity capable to submit jobs to HPC.
    BACKEND_CLIENT = HPCSchedulerClient()
elif TvbProfile.current.cluster.IS_DEPLOY:
    # if TvbProfile.current.cluster.IS_DEPLOY:
    # Return an entity capable to submit jobs to the cluster.
    BACKEND_CLIENT = ClusterSchedulerClient()
else:
    # Return a thread launcher.
    BACKEND_CLIENT = StandAloneClient()
