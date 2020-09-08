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
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

import os
import shutil
import threading
from queue import Queue
from threading import Lock

from syncrypto import Crypto, Syncrypto
from tvb.basic.exceptions import TVBException
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.decorators import synchronized
from tvb.core.entities.file.files_helper import FilesHelper

LOGGER = get_logger(__name__)


class DataEncryptionHandlerMeta(type):
    """
    Metaclass used to generate the singleton instance
    """

    _instance = None

    def __call__(cls):
        if cls._instance is None:
            cls._instance = super(DataEncryptionHandlerMeta, cls).__call__()
        return cls._instance


class DataEncryptionHandler(metaclass=DataEncryptionHandlerMeta):
    CRYPTO_PASS = "CRYPTO_PASS"

    fie_helper = FilesHelper()

    # Queue used to push projects which need synchronisation
    sync_project_queue = Queue()

    # Dict used to count the project frequency from the queue
    queue_elements_count = {}

    # Dict used to count how the project usage
    users_project_usage = {}

    running_operations = {}

    # Dict used to keep projects which are marked for deletion
    marked_for_delete = set()

    # Linked projects
    linked_projects = {}

    lock = Lock()

    def _queue_count(self, folder):
        return self.queue_elements_count[folder] if folder in self.queue_elements_count else 0

    def _project_active_count(self, folder):
        return self.users_project_usage[folder] if folder in self.users_project_usage else 0

    def _running_op_count(self, folder):
        return self.running_operations[folder] if folder in self.running_operations else 0

    @synchronized(lock)
    def inc_project_usage_count(self, folder):
        count = self._project_active_count(folder)
        count += 1
        self.users_project_usage[folder] = count

    @synchronized(lock)
    def dec_project_usage_count(self, folder):
        count = self._project_active_count(folder)
        if count == 1:
            self.users_project_usage.pop(folder)
            return
        count -= 1
        self.users_project_usage[folder] = count

    @synchronized(lock)
    def inc_running_op_count(self, folder):
        count = self._running_op_count(folder)
        count += 1
        self.running_operations[folder] = count

    @synchronized(lock)
    def dec_running_op_count(self, folder):
        count = self._running_op_count(folder)
        if count == 1:
            self.running_operations.pop(folder)
            return
        count -= 1
        self.running_operations[folder] = count

    @synchronized(lock)
    def inc_queue_count(self, folder):
        count = self._queue_count(folder)
        count += 1
        self.queue_elements_count[folder] = count

    @synchronized(lock)
    def dec_queue_count(self, folder):
        count = self._queue_count(folder)
        if count == 1:
            self.queue_elements_count.pop(folder)
            return
        count -= 1
        self.queue_elements_count[folder] = count

    @synchronized(lock)
    def check_and_delete(self, folder):
        # Check if we can remove a folder:
        #   1. It is not in the queue
        #   2. It is marked for delete
        #   3. Nobody is using it
        if self.is_in_usage(folder):
            LOGGER.info("Project {} still in use.".format(folder))
            return
        shutil.rmtree(folder)

    def is_in_usage(self, project_folder):
        return self._queue_count(project_folder) > 0 \
               or self._project_active_count(project_folder) > 0 \
               or self._running_op_count(project_folder) > 0

    @staticmethod
    def compute_encrypted_folder_path(project_folder):
        return "{}_encrypted".format(project_folder)

    @staticmethod
    def sync_folders(folder):
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return
        encrypted_folder = DataEncryptionHandler.compute_encrypted_folder_path(folder)
        crypto_pass = os.environ[
            DataEncryptionHandler.CRYPTO_PASS] if DataEncryptionHandler.CRYPTO_PASS in os.environ else None
        if crypto_pass is None:
            raise TVBException("Storage encryption/decryption is not possible because password is not provided.")
        crypto = Crypto(crypto_pass)
        syncro = Syncrypto(crypto, encrypted_folder, folder)
        syncro.sync_folder()
        trash_path = os.path.join(encrypted_folder, "_syncrypto", "trash")
        if os.path.exists(trash_path):
            shutil.rmtree(trash_path)

    def set_project_active(self, project, linked_dt=None):
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return
        project_folder = self.fie_helper.get_project_folder(project)
        projects = set()
        if linked_dt is None:
            linked_dt = []
        for dt_path in linked_dt:
            project_path = self.fie_helper.get_project_folder_from_h5(dt_path)
            projects.add(project_path)
        if len(linked_dt) > 0:
            self.linked_projects[project_folder] = projects
        projects.add(project_folder)

        for project_folder in projects:
            self.inc_project_usage_count(project_folder)
            self.push_folder_to_sync(project_folder)

    def set_project_inactive(self, project):
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return
        project_folder = self.fie_helper.get_project_folder(project)
        projects = self.linked_projects.pop(project_folder) if project_folder in self.linked_projects else set()
        projects.add(project_folder)
        for project_folder in projects:
            self.dec_project_usage_count(project_folder)
            self.check_and_delete(project_folder)

    def push_folder_to_sync(self, project_folder):
        if not TvbProfile.current.web.ENCRYPT_STORAGE or self._queue_count(project_folder) > 2:
            return
        self.inc_queue_count(project_folder)
        self.sync_project_queue.put(project_folder)


class FoldersQueueConsumer(threading.Thread):
    was_processing = False

    marked_stop = False

    def mark_stop(self):
        self.marked_stop = True

    def run(self):
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return
        while True:
            if encryption_handler.sync_project_queue.empty():
                if self.was_processing:
                    self.was_processing = False
                    LOGGER.info("Finish processing queue")
                if self.marked_stop:
                    break
                continue
            if not self.was_processing:
                LOGGER.info("Start processing queue")
                self.was_processing = True
            folder = encryption_handler.sync_project_queue.get()
            DataEncryptionHandler.sync_folders(folder)
            encryption_handler.dec_queue_count(folder)
            encryption_handler.check_and_delete(folder)
            encryption_handler.sync_project_queue.task_done()


encryption_handler = DataEncryptionHandler()
