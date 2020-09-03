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

import os
import queue
import signal
import sys
from subprocess import Popen, PIPE
from threading import Thread, Event

from tvb.basic.exceptions import TVBException
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcadapter import AdapterLaunchModeEnum
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.model_operation import OperationProcessIdentifier, STATUS_ERROR, STATUS_CANCELED
from tvb.core.entities.storage import dao
from tvb.core.services.backend_clients.backend_client import BackendClient
from tvb.core.services.burst_service import BurstService
from tvb.core.entities.file.data_encryption_handler import encryption_handler

LOGGER = get_logger(__name__)

CURRENT_ACTIVE_THREADS = []

LOCKS_QUEUE = queue.Queue(0)
for i in range(TvbProfile.current.MAX_THREADS_NUMBER):
    LOCKS_QUEUE.put(1)


class OperationExecutor(Thread):
    """
    Thread in charge for starting an operation, used both on cluster and with stand-alone installations.
    """

    def __init__(self, op_id):
        Thread.__init__(self)
        self.operation_id = op_id
        self._stop_ev = Event()

    def run(self):
        """
        Get the required data from the operation queue and launch the operation.
        """
        # Try to get a spot to launch own operation.
        LOCKS_QUEUE.get(True)
        operation_id = self.operation_id
        run_params = [TvbProfile.current.PYTHON_INTERPRETER_PATH, '-m', 'tvb.core.operation_async_launcher',
                      str(operation_id), TvbProfile.CURRENT_PROFILE_NAME]

        current_operation = dao.get_operation_by_id(operation_id)
        project_folder = FilesHelper().get_project_folder(current_operation.project)
        encryption_handler.inc_running_op_count(project_folder)
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

            del launched_process

        encryption_handler.dec_running_op_count(project_folder)
        encryption_handler.check_and_delete(project_folder)

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


class StandAloneClient(BackendClient):
    """
    Instead of communicating with a back-end cluster, fire locally a new thread.
    """

    @staticmethod
    def execute(operation_id, user_name_label, adapter_instance):
        """Start asynchronous operation locally"""
        thread = OperationExecutor(operation_id)
        CURRENT_ACTIVE_THREADS.append(thread)
        if adapter_instance.launch_mode is AdapterLaunchModeEnum.SYNC_DIFF_MEM:
            thread.run()
            operation = dao.get_operation_by_id(operation_id)
            if operation.additional_info and operation.status == STATUS_ERROR:
                raise TVBException(operation.additional_info)
        else:
            thread.start()

    @staticmethod
    def stop_operation(operation_id):
        """
        Stop a thread for a given operation id
        """
        operation = dao.try_get_operation_by_id(operation_id)
        if not operation or operation.has_finished:
            LOGGER.info("Operation already stopped or not found at ID: %s" % operation_id)
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
