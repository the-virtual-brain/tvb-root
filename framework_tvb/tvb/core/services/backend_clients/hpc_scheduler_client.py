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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

import json
import os
import shutil
import typing
import uuid
from contextlib import closing
from enum import Enum
from threading import Thread, Event
from time import sleep

import pyunicore.client as unicore_client
from pyunicore.client import Job, Storage, Client
from requests import HTTPError
from tvb.adapters.datatypes.h5.mapped_value_h5 import DatatypeMeasureH5
from tvb.adapters.simulator.hpc_simulator_adapter import HPCSimulatorAdapter
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter
from tvb.basic.config.settings import HPCSettings
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.simulator_h5 import SimulatorH5
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.model.model_operation import has_finished, STATUS_FINISHED, Operation, STATUS_CANCELED, \
    STATUS_ERROR, OperationProcessIdentifier
from tvb.core.entities.storage import dao, OperationDAO
from tvb.core.neocom import h5
from tvb.core.neotraits.h5 import H5File
from tvb.core.services.backend_clients import secure_data_store_copy as secure_data_store
from tvb.core.services.backend_clients.backend_client import BackendClient
from tvb.core.services.burst_service import BurstService

LOGGER = get_logger(__name__)

HPC_THREADS = []


class HPCJobStatus(Enum):
    STAGINGIN = "STAGINGIN"
    READY = "READY"
    QUEUED = "QUEUED"
    STAGINGOUT = "STAGINGOUT"
    SUCCESSFUL = "SUCCESSFUL"
    FAILED = "FAILED"


def get_op_thread(op_id):
    # type: (int) -> HPCOperationThread
    op_thread = None
    for thread in HPC_THREADS:
        if thread.operation_id == op_id:
            op_thread = thread
            break
    if op_thread is not None:
        HPC_THREADS.remove(op_thread)
    return op_thread


class HPCOperationThread(Thread):
    def __init__(self, operation_id, *args, **kwargs):
        super(HPCOperationThread, self).__init__(*args, **kwargs)
        self.operation_id = operation_id
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class EncryptionHandler(object):
    encrypted_dir_name_regex = 'cipher_{}'

    def __init__(self, dir_gid, config_path=TvbProfile.current.hpc.SDS_CONFIG_PATH):
        """
        :param dir_gid: the GID to use for the encrypted directory name
        :param config_path: path towards the secure_data_store configuration file (usually called .sdsrd)
        """
        self.set_dir_gid(dir_gid)
        self.config = self._load_configuration(config_path)

    def _prepare_encrypted_dir_name(self):
        return self.encrypted_dir_name_regex.format(self.dir_gid)

    def _load_configuration(self, config_path):
        config = secure_data_store.read_config(config_path)
        return config

    def set_dir_gid(self, dir_gid):
        if isinstance(dir_gid, uuid.UUID):
            dir_gid = dir_gid.hex
        self.dir_gid = dir_gid
        self.encrypted_dir_name = self._prepare_encrypted_dir_name()

    def get_encrypted_dir(self):
        return secure_data_store.datastore(self.config, self.encrypted_dir_name)

    def get_passfile(self):
        return secure_data_store.passstore(self.config, self.encrypted_dir_name)

    def prepare_encryption_dir(self):
        secure_data_store.create(self.config, self.encrypted_dir_name)
        encrypted_dir = self.get_encrypted_dir()
        return encrypted_dir

    def open_plain_dir(self):
        # ! Always call close_plain_dir() after this method !
        plain_dir = secure_data_store.mount(self.config, self.encrypted_dir_name)
        return plain_dir

    def close_plain_dir(self):
        secure_data_store.unmount(self.config, self.encrypted_dir_name)

    def encrypt_inputs(self, files_to_encrypt):
        # type: (list) -> list
        """
        Receive a list with all files to encrypt.
        Prepare encryption directory and mount a plain directory to it.
        Copy all files to the plain directory and unmount it.
        Return a list with all files from the encrypted directory.
        """
        encryption_dir = self.prepare_encryption_dir()
        plain_dir = self.open_plain_dir()

        for file_to_encrypt in files_to_encrypt:
            shutil.copy(file_to_encrypt, plain_dir)
        self.close_plain_dir()

        encrypted_files = [os.path.join(encryption_dir, enc_file) for enc_file in os.listdir(encryption_dir)]
        return encrypted_files

    def decrypt_results_to_dir(self, dir):
        # type: (str) -> None
        """
        Having an already encrypted directory, mount a plain directory to it and decrypt files,
        then move plain files to the location specified by :param dir
        """
        plain_dir = self.open_plain_dir()
        for plain_file in os.listdir(plain_dir):
            shutil.copy(os.path.join(plain_dir, plain_file), dir)
        self.close_plain_dir()


class HPCSchedulerClient(BackendClient):
    """
    Simple class, to mimic the same behavior we are expecting from StandAloneClient, but firing the operation on
    an HPC node. Define TVB_BIN_ENV_KEY and CSCS_LOGIN_TOKEN_ENV_KEY as environment variables before running on HPC.
    """
    TVB_BIN_ENV_KEY = 'TVB_BIN'
    CSCS_LOGIN_TOKEN_ENV_KEY = 'CSCS_LOGIN_TOKEN'
    file_handler = FilesHelper()

    @staticmethod
    def _gather_file_list(base_h5_file, file_list):
        # type: (H5File, list) -> None
        references = base_h5_file.gather_references_gids()
        for reference in references:
            if reference is None:
                continue
            try:
                # TODO: nicer way to identify files?
                index = dao.get_datatype_by_gid(reference.hex)
                reference_h5_path = h5.path_for_stored_index(index)
                reference_h5_file = h5.h5_file_for_index(index)
            except Exception:
                reference_h5_path = base_h5_file.get_reference_path(reference)
                reference_h5_file = H5File.from_file(reference_h5_path)
            if reference_h5_path not in file_list:
                file_list.append(reference_h5_path)
            HPCSchedulerClient._gather_file_list(reference_h5_file, file_list)
            reference_h5_file.close()

    @staticmethod
    def _prepare_input(operation, simulator_gid):
        # type: (Operation, str) -> list
        storage_path = FilesHelper().get_project_folder(operation.project, str(operation.id))
        simulator_in_path = h5.path_for(storage_path, SimulatorH5, simulator_gid)

        input_files_list = []
        with SimulatorH5(simulator_in_path) as simulator_h5:
            HPCSchedulerClient._gather_file_list(simulator_h5, input_files_list)
        input_files_list.append(simulator_in_path)
        return input_files_list

    @staticmethod
    def _configure_job(simulator_gid, available_space, is_group_launch):
        # type: (str, int, bool) -> (dict, list)
        bash_entrypoint = os.path.join(os.environ[HPCSchedulerClient.TVB_BIN_ENV_KEY],
                                       HPCSettings.HPC_LAUNCHER_SH_SCRIPT)
        base_url = TvbProfile.current.web.BASE_URL
        inputs_in_container = os.path.join('/root/sds/containers', EncryptionHandler(simulator_gid).encrypted_dir_name,
                                           'FS')

        # Build job configuration JSON
        my_job = {}
        my_job[HPCSettings.UNICORE_EXE_KEY] = os.path.basename(bash_entrypoint)
        my_job[HPCSettings.UNICORE_ARGS_KEY] = [simulator_gid, available_space, is_group_launch, base_url,
                                                inputs_in_container]
        my_job[HPCSettings.UNICORE_RESOURCER_KEY] = {"CPUs": "1"}

        return my_job, bash_entrypoint

    @staticmethod
    def listdir(working_dir, base='/'):
        # type: (Storage, str) -> dict
        """
        We took this code from pyunicore Storage.listdir method and extended it to use a subdirectory.
        Looking at the method signature, it should have had this behavior, but the 'base' argument is not used later
        inside the method code.
        TODO: Probably will be fixed soon in their API, so we could delete this.
        :return: dict of {str: PathFile} objects
        """
        ret = {}
        for path, meta in working_dir.contents(base)['content'].items():
            path_url = working_dir.path_urls['files'] + path
            path = path[1:]  # strip leading '/'
            if meta['isDirectory']:
                ret[path] = unicore_client.PathDir(working_dir, path_url, path)
            else:
                ret[path] = unicore_client.PathFile(working_dir, path_url, path)
        return ret

    def update_db_with_results(self, operation, result_filenames):
        # type: (Operation, list) -> (str, int)
        """
        Generate corresponding Index entities for the resulted H5 files and insert them in DB.
        """
        burst_service = BurstService()
        index_list = []

        LOGGER.info("Marking operation {} as finished...".format(operation.id))
        operation.mark_complete(STATUS_FINISHED)
        dao.store_entity(operation)

        for filename in result_filenames:
            index = h5.index_for_h5_file(filename)()
            # TODO: don't load full TS in memory and make this read nicer
            try:
                datatype, ga = h5.load_with_references(filename)
                index.fill_from_has_traits(datatype)
                index.fill_from_generic_attributes(ga)
            except TypeError:
                with DatatypeMeasureH5(filename) as dti_h5:
                    index.metrics = json.dumps(dti_h5.metrics.load())
                    index.source_gid = dti_h5.analyzed_datatype.load().hex
            index.fk_from_operation = operation.id
            if operation.fk_operation_group:
                datatype_group = \
                    dao.get_generic_entity(DataTypeGroup, operation.fk_operation_group, 'fk_operation_group')[0]
                index.fk_datatype_group = datatype_group.id

            # TODO: update status during operation run
            if operation.fk_operation_group:
                parent_burst = \
                    dao.get_generic_entity(BurstConfiguration, operation.fk_operation_group, 'operation_group_id')[0]
                operations_in_group = dao.get_operations_in_group(operation.fk_operation_group)
                if parent_burst.metric_operation_group_id:
                    operations_in_group.extend(dao.get_operations_in_group(parent_burst.metric_operation_group_id))
                for operation in operations_in_group:
                    if not has_finished(operation.status):
                        break
                    if parent_burst is not None:
                        burst_service.mark_burst_finished(parent_burst)
                        index.fk_parent_burst = parent_burst.id
            else:
                parent_burst = burst_service.get_burst_for_operation_id(operation.id)
                if parent_burst is not None:
                    burst_service.mark_burst_finished(parent_burst)
                    index.fk_parent_burst = parent_burst.id

            index = dao.store_entity(index)
            index_list.append(index)

        sim_adapter = SimulatorAdapter()
        mesage, _ = sim_adapter.fill_existing_indexes(operation, index_list)

        # TODO: for PSE set FK towards datatype group on results
        return mesage

    @staticmethod
    def _create_job_with_pyunicore(pyunicore_client, job_description, inputs=[]):
        # type: (Client, {}, list) -> Job
        """
        Submit and start a batch job on the site, optionally uploading input data files.
        We took this code from the pyunicore Client.new_job method in order to use our own upload method
        :return: job
        """

        if len(inputs) > 0 or job_description.get('haveClientStageIn') is True:
            job_description['haveClientStageIn'] = "true"

        with closing(
                pyunicore_client.transport.post(url=pyunicore_client.site_urls['jobs'], json=job_description)) as resp:
            job_url = resp.headers['Location']

        job = Job(pyunicore_client.transport, job_url)

        if len(inputs) > 0:
            working_dir = job.working_dir
            for input in inputs:
                HPCSchedulerClient._upload_file_with_pyunicore(working_dir, input)
        if job_description.get('haveClientStageIn', None) == "true":
            try:
                job.start()
            except:
                pass

        return job

    @staticmethod
    def _upload_file_with_pyunicore(working_dir, input_name, destination=None):
        # type: (Storage, str, str) -> None
        """
        Upload file to the HPC working dir.
        We took this upload code from pyunicore Storage.upload method and modified it because in the original code the
        upload URL is generated using the os.path.join method. The result is an invalid URL for windows os.
        """
        if destination is None:
            destination = os.path.basename(input_name)

        headers = {'Content-Type': 'application/octet-stream'}
        with open(input_name, 'rb') as fd:
            working_dir.transport.put(
                url="{}/{}/{}".format(working_dir.resource_url, "files", destination),
                headers=headers,
                data=fd)

    @staticmethod
    def _build_unicore_client(auth_token, registry_url, supercomputer):
        # type: (str, str, str) -> Client
        transport = unicore_client.Transport(auth_token)
        registry = unicore_client.Registry(transport, registry_url)
        return registry.site(supercomputer)

    @staticmethod
    def _launch_job_with_pyunicore(operation, simulator_gid, is_group_launch):
        # type: (Operation, str, bool) -> Job
        LOGGER.info("Prepare job inputs for operation: {}".format(operation.id))
        job_inputs = HPCSchedulerClient._prepare_input(operation, simulator_gid)
        available_space = HPCSchedulerClient.compute_available_disk_space(operation)

        LOGGER.info("Prepare job configuration for operation: {}".format(operation.id))
        job_config, job_script = HPCSchedulerClient._configure_job(simulator_gid, available_space,
                                                                   is_group_launch)

        LOGGER.info("Prepare encryption for operation: {}".format(operation.id))
        encryption_handler = EncryptionHandler(simulator_gid)
        LOGGER.info("Encrypt job inputs for operation: {}".format(operation.id))
        job_encrypted_inputs = encryption_handler.encrypt_inputs(job_inputs)
        encrypted_dir = encryption_handler.get_encrypted_dir()
        script_path = shutil.copy(job_script, encrypted_dir)
        job_encrypted_inputs.append(script_path)

        # use "DAINT-CSCS" -- change if another supercomputer is prepared for usage
        LOGGER.info("Prepare unicore client for operation: {}".format(operation.id))
        site_client = HPCSchedulerClient._build_unicore_client(os.environ[HPCSchedulerClient.CSCS_LOGIN_TOKEN_ENV_KEY],
                                                               unicore_client._HBP_REGISTRY_URL,
                                                               TvbProfile.current.hpc.HPC_COMPUTE_SITE)

        LOGGER.info("Submit job for operation: {}".format(operation.id))
        job = HPCSchedulerClient._create_job_with_pyunicore(pyunicore_client=site_client, job_description=job_config,
                                                            inputs=job_encrypted_inputs)
        LOGGER.info("Job url {} for operation: {}".format(job.resource_url, operation.id))
        op_identifier = OperationProcessIdentifier(operation_id=operation.id, job_id=job.resource_url)
        dao.store_entity(op_identifier)
        LOGGER.info("Job mount point: {}".format(job.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]))
        return job

    @staticmethod
    def compute_available_disk_space(operation):
        # type: (Operation) -> int
        disk_space_per_user = TvbProfile.current.MAX_DISK_SPACE
        pending_op_disk_space = dao.compute_disk_size_for_started_ops(operation.fk_launched_by)
        user_disk_space = dao.compute_user_generated_disk_size(operation.fk_launched_by)  # From kB to Bytes
        available_space = disk_space_per_user - pending_op_disk_space - user_disk_space
        return available_space

    @staticmethod
    # TODO: Remove this
    def _monitor_job(job):
        # type: (Job) -> (Storage, str)
        LOGGER.info(job.properties)
        # TODO: better monitoring?
        LOGGER.info("Job mount point: {}".format(job.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]))
        files_encryption_key_uploaded = False
        sleep_period = 5
        while True:
            # Read current job status
            status = job.properties[HPCSettings.JOB_STATUS_KEY]

            # Check if the current job is complete
            if status in (HPCJobStatus.SUCCESSFUL.value, HPCJobStatus.FAILED.value):
                break
            LOGGER.info("Job is still running. Status: {}".format(status))

            # Upload user tokens which will be used to
            if status == HPCJobStatus.READY.value and not files_encryption_key_uploaded:
                LOGGER.info("Upload files encryption key")
                # HPCSchedulerClient._upload_file_with_pyunicore(job.working_dir, "Files encryption key path")
                files_encryption_key_uploaded = True
                sleep_period = 10
            sleep(sleep_period)

        LOGGER.info("Job finished with status: {}".format(status))
        return job.working_dir, status

    @staticmethod
    def stage_out_to_operation_folder(working_dir, operation, simulator_gid):
        # type: (Storage, Operation, typing.Union[uuid.UUID, str]) -> list
        output_list = HPCSchedulerClient.listdir(working_dir, HPCSimulatorAdapter.OUTPUT_FOLDER)
        encryption_handler = EncryptionHandler(simulator_gid)
        # TODO: output folder name is encrypted too
        encrypted_dir = os.path.join(encryption_handler.config.dataroot, HPCSimulatorAdapter.OUTPUT_FOLDER)
        HPCSchedulerClient._stage_out_outputs(encrypted_dir, output_list)

        operation_dir = HPCSchedulerClient.file_handler.get_project_folder(operation.project, str(operation.id))
        encryption_handler.decrypt_results_to_dir(operation_dir)
        h5_filenames = [os.path.join(operation_dir, h5_file) for h5_file in os.listdir(operation_dir)]

        LOGGER.info(working_dir.properties)
        LOGGER.info(working_dir.listdir())
        return h5_filenames

    @staticmethod
    def _run_hpc_job(operation_identifier):
        # type: (int) -> None
        operation = dao.get_operation_by_id(operation_identifier)
        is_group_launch = operation.fk_operation_group is not None
        simulator_gid = json.loads(operation.parameters)['gid']
        try:
            HPCSchedulerClient._launch_job_with_pyunicore(operation, simulator_gid, is_group_launch)
        except Exception as exception:
            LOGGER.error("Failed to submit job HPC", exc_info=True)
            operation.mark_complete(STATUS_ERROR,
                                    exception.response.text if isinstance(exception, HTTPError) else repr(exception))
            dao.store_entity(operation)

    @staticmethod
    def _stage_out_outputs(encrypted_dir_path, output_list):
        # type: (str, dict) -> None
        for output_filename, output_filepath in output_list.items():
            if type(output_filepath) is not unicore_client.PathFile:
                continue
            filename = os.path.join(encrypted_dir_path, os.path.basename(output_filename))
            output_filepath.download(filename)

    @staticmethod
    def execute(operation_id, user_name_label, adapter_instance):
        # type: (int, None, None) -> None
        """Call the correct system command to submit a job to HPC."""
        thread = HPCOperationThread(operation_id, target=HPCSchedulerClient._run_hpc_job,
                                    kwargs={'operation_identifier': operation_id})
        thread.start()
        HPC_THREADS.append(thread)

    @staticmethod
    def stop_operation(operation_id):
        # TODO: Review this implementation after DAINT maintenance
        operation = dao.get_operation_by_id(operation_id)
        if not operation or operation.has_finished:
            LOGGER.warning("Operation already stopped: %s" % operation_id)
            return True

        LOGGER.debug("Stopping HPC operation: %s" % str(operation_id))
        op_ident = OperationDAO().get_operation_process_for_operation(operation_id)
        if op_ident is not None:
            # TODO: Handle login
            transport = unicore_client.Transport(os.environ[HPCSchedulerClient.CSCS_LOGIN_TOKEN_ENV_KEY])
            # Abort HPC job
            job = Job(transport, op_ident.job_id)
            if job.is_running():
                job.abort()

        # Kill thread
        operation_thread = get_op_thread(operation_id)
        if operation_thread is None:
            LOGGER.warning("Thread for operation {} is not available".format(operation_id))
        else:
            operation_thread.stop()
            while not operation_thread.stopped():
                LOGGER.info("Thread for operation {} is stopping".format(operation_id))
        BurstService().persist_operation_state(operation, STATUS_CANCELED)
        return True
