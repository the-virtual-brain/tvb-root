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
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

import os
import shutil
import threading
from io import BytesIO
from os import stat
from queue import Queue
from threading import Lock

import pyAesCrypt
import requests
from tvb.basic.exceptions import TVBException
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.storage.h5.decorators import synchronized
from tvb.storage.h5.encryption.encryption_handler import EncryptionHandler
from tvb.storage.h5.file.files_helper import FilesHelper
from tvb.storage.kube.kube_notifier import KubeNotifier

LOGGER = get_logger(__name__)

try:
    from syncrypto import Crypto, Syncrypto
except ModuleNotFoundError:
    LOGGER.info("Cannot import syncrypto library.")


class InvalidStorageEncryptionException(TVBException):
    """
    Exception thrown when encryption storage cannot be allowed.
    """

    def __init__(self, message):
        TVBException.__init__(self, message)


class DataEncryptionHandlerMeta(type):
    """
    Metaclass used to generate the singleton instance
    """

    _instances = {}

    def __call__(cls):
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = super(DataEncryptionHandlerMeta, cls).__call__()
        return DataEncryptionHandlerMeta._instances[cls]


class DataEncryptionHandler(metaclass=DataEncryptionHandlerMeta):
    ENCRYPTED_FOLDER_SUFFIX = "_encrypted"
    KEYS_FOLDER = ".storage-keys"
    CRYPTO_PASS = "CRYPTO_PASS"
    APP_ENCRYPTION_HANDLER = "APP_ENCRYPTION_HANDLER"

    file_helper = FilesHelper()

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
    def _dec_project_usage_count(self, folder):
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
    def _inc_queue_count(self, folder):
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
        if not self.is_in_usage(folder) \
                and folder in self.marked_for_delete:
            self.marked_for_delete.remove(folder)
            LOGGER.info("Remove folder {}".format(folder))
            shutil.rmtree(folder)

    def is_in_usage(self, folder):
        return self._queue_count(folder) > 0 \
               or self._project_active_count(folder) > 0 \
               or self._running_op_count(folder) > 0

    @staticmethod
    def compute_encrypted_folder_path(current_project_folder):
        project_name = os.path.basename(current_project_folder)
        project_path = os.path.join(TvbProfile.current.TVB_STORAGE, FilesHelper.PROJECTS_FOLDER, project_name)
        return "{}{}".format(project_path, DataEncryptionHandler.ENCRYPTED_FOLDER_SUFFIX)

    def sync_folders(self, folder):
        if not self.encryption_enabled():
            return

        project_name = os.path.basename(folder)
        encrypted_folder = self.compute_encrypted_folder_path(folder)

        if os.path.exists(encrypted_folder) or os.path.exists(folder):
            crypto_pass = self._project_key(project_name)
            crypto = Crypto(crypto_pass)
            syncro = Syncrypto(crypto, encrypted_folder, folder)
            syncro.sync_folder()
            trash_path = os.path.join(encrypted_folder, "_syncrypto", "trash")
            if os.path.exists(trash_path):
                shutil.rmtree(trash_path)
        else:
            LOGGER.info("Project {} was deleted".format(project_name))

    def set_project_active(self, project, linked_dt):
        if not self.encryption_enabled():
            return
        project_folder = self.file_helper.get_project_folder(project.name)
        projects = set()
        if linked_dt is None:
            linked_dt = []
        for dt_path in linked_dt:
            project_path = FilesHelper.get_project_folder_from_h5(dt_path)
            projects.add(project_path)
        if len(linked_dt) > 0:
            self.linked_projects[project_folder] = projects
        projects.add(project_folder)

        for project_folder in projects:
            self.inc_project_usage_count(project_folder)
            self.push_folder_to_sync(project_folder)

    def set_project_inactive(self, project):
        if not self.encryption_enabled():
            return
        project_folder = self.file_helper.get_project_folder(project)
        projects = self.linked_projects.pop(project_folder) if project_folder in self.linked_projects else set()
        projects.add(project_folder)
        for project_folder in projects:
            self._dec_project_usage_count(project_folder)
            if self._queue_count(project_folder) > 0 \
                    or self._project_active_count(project_folder) > 0 \
                    or self._running_op_count(project_folder) > 0:
                self.marked_for_delete.add(project_folder)
                LOGGER.info("Project {} still in use. Marked for deletion.".format(project_folder))
                continue
            LOGGER.info("Remove project: {}".format(project_folder))
            shutil.rmtree(project_folder)

    def push_folder_to_sync(self, folder):
        if not self.encryption_enabled() or self._queue_count(folder) > 0:
            return
        self._inc_queue_count(folder)
        self.sync_project_queue.put(folder)

    @staticmethod
    def encryption_enabled():
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return False
        if not TvbProfile.current.web.CAN_ENCRYPT_STORAGE:
            raise InvalidStorageEncryptionException(
                "We can not enable STORAGE ENCRYPTION. Most probably syncrypto is not installed!")
        return True

    @staticmethod
    def app_encryption_handler():
        app_encryption_handler = True if DataEncryptionHandler.APP_ENCRYPTION_HANDLER in os.environ and os.environ[
            DataEncryptionHandler.APP_ENCRYPTION_HANDLER].lower() == 'true' else False
        return not TvbProfile.current.web.OPENSHIFT_DEPLOY or app_encryption_handler

    @staticmethod
    def _get_unencrypted_projects(projects_folder):
        project_list = os.listdir(projects_folder)
        return list(map(lambda project: os.path.join(projects_folder, str(project)),
                        filter(lambda project: not str(project).endswith(DataEncryptionHandler.ENCRYPTED_FOLDER_SUFFIX),
                               project_list)))

    def startup_cleanup(self):
        unencrypted_projects_folder = os.path.join(TvbProfile.current.TVB_STORAGE, FilesHelper.PROJECTS_FOLDER)
        projects_list = self._get_unencrypted_projects(unencrypted_projects_folder)
        for project in projects_list:
            LOGGER.info("Sync and clean project: {}".format(project))
            self.sync_folders(project)
            shutil.rmtree(project)

    @staticmethod
    def project_key_path(project_name):
        return os.path.join(TvbProfile.current.TVB_STORAGE, DataEncryptionHandler.KEYS_FOLDER, str(project_name))

    @staticmethod
    def _project_key(project_name):
        password_encryption_key = os.environ[
            DataEncryptionHandler.CRYPTO_PASS] if DataEncryptionHandler.CRYPTO_PASS in os.environ else None
        if password_encryption_key is None:
            raise TVBException("Password encryption key is not defined.")
        project_keys_folder = os.path.join(TvbProfile.current.TVB_STORAGE, DataEncryptionHandler.KEYS_FOLDER)
        DataEncryptionHandler.file_helper.check_created(project_keys_folder)

        encrypted_project_key = DataEncryptionHandler.project_key_path(project_name)
        if os.path.exists(encrypted_project_key):
            with open(encrypted_project_key, "rb") as fIn:
                inputFileSize = stat(encrypted_project_key).st_size
                pass_stream = BytesIO()
                pyAesCrypt.decryptStream(fIn, pass_stream, password_encryption_key, 64 * 1024, inputFileSize)
                project_key = pass_stream.getvalue().decode()
                pass_stream.close()

            return project_key

        project_key = EncryptionHandler.generate_random_password()
        with open(encrypted_project_key, "wb") as fOut:
            pass_stream = BytesIO(str.encode(project_key))
            pyAesCrypt.encryptStream(pass_stream, fOut, password_encryption_key, 64 * 1024)
            pass_stream.close()
        return project_key


class FoldersQueueConsumer(threading.Thread):
    was_processing = False

    marked_stop = False

    def mark_stop(self):
        self.marked_stop = True

    def run(self):
        if not encryption_handler.encryption_enabled():
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
            encryption_handler.sync_folders(folder)
            encryption_handler.dec_queue_count(folder)
            encryption_handler.check_and_delete(folder)
            encryption_handler.sync_project_queue.task_done()


class DataEncryptionRemoteHandler(DataEncryptionHandler):
    lock = Lock()

    @staticmethod
    def _notify_pods(method, folder=None, **kwargs):
        if folder:
            kwargs['folder'] = folder
        if TvbProfile.current.web.OPENSHIFT_DATA_ENCRYPTION_HANDLER_APPLICATION == "":
            raise TVBException("Openshift Data Encryption handler application is not defined")
        openshift_pods = KubeNotifier.get_pods(TvbProfile.current.web.OPENSHIFT_DATA_ENCRYPTION_HANDLER_APPLICATION)
        if len(openshift_pods) == 0:
            raise TVBException("Openshift Data Encryption handler app not found")
        encryption_app = openshift_pods[0]
        auth_header = KubeNotifier.get_authorization_header()
        url = "http://{}:{}/kube/data_encryption_handler/{}".format(encryption_app.ip, str(TvbProfile.current.web.SERVER_PORT), method)
        return requests.post(url=url, headers=auth_header, data=kwargs)

    @synchronized(lock)
    def inc_project_usage_count(self, folder):
        self._notify_pods("inc_project_usage_count", folder)

    @synchronized(lock)
    def check_and_delete(self, folder):
        self._notify_pods("check_and_delete", folder)

    @synchronized(lock)
    def dec_queue_count(self, folder):
        self._notify_pods("dec_queue_count", folder)

    @synchronized(lock)
    def dec_running_op_count(self, folder):
        self._notify_pods("dec_running_op_count", folder)

    @synchronized(lock)
    def inc_project_usage_count(self, folder):
        self._notify_pods("inc_project_usage_count", folder)

    @synchronized(lock)
    def inc_running_op_count(self, folder):
        self._notify_pods("inc_running_op_count", folder)

    @synchronized(lock)
    def is_in_usage(self, folder):
        response = self._notify_pods("is_in_usage", folder)
        in_usage = True
        if response.ok and response.content.decode('utf-8').lower() == 'false':
            in_usage = False
        return in_usage

    def push_folder_to_sync(self, folder):
        self._notify_pods("push_folder_to_sync", folder)

    def startup_cleanup(self):
        self._notify_pods("startup_cleanup")

    def sync_folders(self, folder):
        self._notify_pods("sync_folders", folder)

    def set_project_active(self, project, linked_dt):
        self._notify_pods("set_project_active", **{'project': project, 'linked_dt': linked_dt})

    def set_project_inactive(self, project):
        self._notify_pods("set_project_inactive", **{'project': project})


class DataEncryptionHandlerBuilder:
    @staticmethod
    def build_handler():
        if DataEncryptionHandler.app_encryption_handler():
            return DataEncryptionHandler()
        return DataEncryptionRemoteHandler()


encryption_handler = DataEncryptionHandlerBuilder.build_handler()
