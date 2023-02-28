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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

import os
import typing
import uuid
from contextlib import closing
from enum import Enum
from threading import Thread, Event
from requests import HTTPError

from tvb.basic.config.settings import HPCSettings
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config import MEASURE_METRICS_MODEL_CLASS
from tvb.core.entities.file.simulator.datatype_measure_h5 import DatatypeMeasureH5
from tvb.core.entities.model.model_operation import Operation, STATUS_CANCELED, STATUS_ERROR, OperationProcessIdentifier
from tvb.core.entities.storage import dao, OperationDAO
from tvb.core.neocom import h5
from tvb.core.services.backend_clients.backend_client import BackendClient
from tvb.core.services.burst_service import BurstService
from tvb.core.services.exceptions import OperationException
from tvb.storage.storage_interface import StorageInterface

try:
    import pyunicore.client as unicore_client
    from pyunicore.client import Job, Storage, Client
except ImportError:
    HPCSettings.CAN_RUN_HPC = False

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


class HPCSchedulerClient(BackendClient):
    """
    Simple class, to mimic the same behavior we are expecting from StandAloneClient, but firing the operation on
    an HPC node. Define TVB_BIN_ENV_KEY and CSCS_LOGIN_TOKEN_ENV_KEY as environment variables before running on HPC.
    """
    OUTPUT_FOLDER = 'output'
    TVB_BIN_ENV_KEY = 'TVB_BIN'
    CSCS_LOGIN_TOKEN_ENV_KEY = 'CSCS_LOGIN_TOKEN'
    CSCS_PROJECT = 'CSCS_PROJECT'
    HOME_FOLDER_MOUNT = '/HOME_FOLDER'
    CSCS_DATA_FOLDER = 'data'
    CONTAINER_INPUT_FOLDER = '/home/tvb_user/.data'
    storage_interface = StorageInterface()

    @staticmethod
    def _prepare_input(operation, simulator_gid):
        # type: (Operation, str) -> list
        storage_path = StorageInterface().get_project_folder(operation.project.name,
                                                             str(operation.id))
        vm_files, dt_files = h5.gather_references_of_view_model(simulator_gid, storage_path)
        vm_files.extend(dt_files)
        return vm_files

    @staticmethod
    def _configure_job(simulator_gid, available_space, is_group_launch, operation_id):
        # type: (str, int, bool, int) -> (dict, list)
        bash_entrypoint = os.path.join(os.environ[HPCSchedulerClient.TVB_BIN_ENV_KEY],
                                       HPCSettings.HPC_LAUNCHER_SH_SCRIPT)
        base_url = TvbProfile.current.web.BASE_URL
        inputs_in_container = os.path.join(
            HPCSchedulerClient.CONTAINER_INPUT_FOLDER,
            StorageInterface.get_encryption_handler(simulator_gid).current_enc_dirname)

        # Build job configuration JSON
        my_job = {HPCSettings.UNICORE_EXE_KEY: os.path.basename(bash_entrypoint),
                  HPCSettings.UNICORE_ARGS_KEY: [simulator_gid, available_space, is_group_launch, base_url,
                                                 inputs_in_container, HPCSchedulerClient.HOME_FOLDER_MOUNT,
                                                 operation_id], HPCSettings.UNICORE_RESOURCER_KEY: {"CPUs": "1"}}

        if HPCSchedulerClient.CSCS_PROJECT in os.environ:
            my_job[HPCSettings.UNICORE_PROJECT_KEY] = os.environ[HPCSchedulerClient.CSCS_PROJECT]

        return my_job, bash_entrypoint

    @staticmethod
    def _listdir(working_dir, base='/'):
        # type: (Storage, str) -> dict
        """
        We took this code from pyunicore Storage.listdir method and extended it to use a subdirectory.
        Looking at the method signature, it should have had this behavior, but the 'base' argument is not used later
        inside the method code.
        Probably will be fixed soon in their API, so we could delete this.
        :return: dict of {str: PathFile} objects
        """
        ret = {}
        try:
            for path, meta in working_dir.contents(base)['content'].items():
                path_url = working_dir.path_urls['files'] + path
                path = path[1:]  # strip leading '/'
                if meta['isDirectory']:
                    ret[path] = unicore_client.PathDir(working_dir, path_url, path)
                else:
                    ret[path] = unicore_client.PathFile(working_dir, path_url, path)
            return ret
        except HTTPError as http_error:
            if http_error.response.status_code == 404:
                raise OperationException("Folder {} is not present on HPC storage.".format(base))
            raise http_error

    def update_datatype_groups(self):
        # TODO: update column count_results
        pass

    @staticmethod
    def update_db_with_results(operation, sim_h5_filenames, metric_operation, metric_h5_filename):
        # type: (Operation, list, Operation, str) -> (str, int)
        """
        Generate corresponding Index entities for the resulted H5 files and insert them in DB.
        """
        burst_service = BurstService()
        index_list = []
        is_group = operation.fk_operation_group is not None
        burst_config = burst_service.get_burst_for_operation_id(operation.id)
        if is_group:
            burst_config = burst_service.get_burst_for_operation_id(operation.fk_operation_group, True)
        all_indexes = burst_service.prepare_indexes_for_simulation_results(operation, sim_h5_filenames, burst_config)
        if is_group:
            # Update the operation group name
            operation_group = dao.get_operationgroup_by_id(metric_operation.fk_operation_group)
            operation_group.fill_operationgroup_name("DatatypeMeasureIndex")
            dao.store_entity(operation_group)

            metric_index = burst_service.prepare_index_for_metric_result(metric_operation, metric_h5_filename,
                                                                         burst_config)
            all_indexes.append(metric_index)

        for index in all_indexes:
            index = dao.store_entity(index)
            index_list.append(index)

        burst_service.update_burst_status(burst_config)

    @staticmethod
    def _create_job_with_pyunicore(pyunicore_client, job_description, job_script, inputs):
        # type: (Client, {}, str, list) -> Job
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
            HPCSchedulerClient._upload_file_with_pyunicore(working_dir, job_script, None)
            for input in inputs:
                HPCSchedulerClient._upload_file_with_pyunicore(working_dir, input)
        if job_description.get('haveClientStageIn', None) == "true":
            try:
                job.start()
            except:
                pass

        return job

    @staticmethod
    def _upload_file_with_pyunicore(working_dir, input_name, subfolder=CSCS_DATA_FOLDER, destination=None):
        # type: (Storage, str, object, str) -> None
        """
        Upload file to the HPC working dir.
        We took this upload code from pyunicore Storage.upload method and modified it because in the original code the
        upload URL is generated using the os.path.join method. The result is an invalid URL for windows os.
        """
        if destination is None:
            destination = os.path.basename(input_name)

        if subfolder:
            url = "{}/{}/{}/{}".format(working_dir.resource_url, "files", subfolder, destination)
        else:
            url = "{}/{}/{}".format(working_dir.resource_url, "files", destination)

        headers = {'Content-Type': 'application/octet-stream'}
        with open(input_name, 'rb') as fd:
            working_dir.transport.put(
                url=url,
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
        job_plain_inputs = HPCSchedulerClient._prepare_input(operation, simulator_gid)
        available_space = HPCSchedulerClient.compute_available_disk_space(operation)

        LOGGER.info("Prepare job configuration for operation: {}".format(operation.id))
        job_config, job_script = HPCSchedulerClient._configure_job(simulator_gid, available_space,
                                                                   is_group_launch, operation.id)

        LOGGER.info("Prepare encryption for operation: {}".format(operation.id))
        encryption_handler = StorageInterface.get_encryption_handler(simulator_gid)
        LOGGER.info("Encrypt job inputs for operation: {}".format(operation.id))
        job_encrypted_inputs = encryption_handler.encrypt_inputs(job_plain_inputs)

        # use "DAINT-CSCS" -- change if another supercomputer is prepared for usage
        LOGGER.info("Prepare unicore client for operation: {}".format(operation.id))
        site_client = HPCSchedulerClient._build_unicore_client(os.environ[HPCSchedulerClient.CSCS_LOGIN_TOKEN_ENV_KEY],
                                                               unicore_client._HBP_REGISTRY_URL,
                                                               TvbProfile.current.hpc.HPC_COMPUTE_SITE)

        LOGGER.info("Submit job for operation: {}".format(operation.id))
        job = HPCSchedulerClient._create_job_with_pyunicore(pyunicore_client=site_client, job_description=job_config,
                                                            job_script=job_script, inputs=job_encrypted_inputs)
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
    def _stage_out_results(working_dir, simulator_gid):
        # type: (Storage, typing.Union[uuid.UUID, str]) -> list
        output_subfolder = HPCSchedulerClient.CSCS_DATA_FOLDER + '/' + HPCSchedulerClient.OUTPUT_FOLDER
        output_list = HPCSchedulerClient._listdir(working_dir, output_subfolder)
        LOGGER.info("Output list {}".format(output_list))
        storage_interface = StorageInterface()
        encrypted_dir = os.path.join(storage_interface.get_encryption_handler(simulator_gid).get_encrypted_dir(),
                                     HPCSchedulerClient.OUTPUT_FOLDER)
        encrypted_files = HPCSchedulerClient._stage_out_outputs(encrypted_dir, output_list)

        # Clean data uploaded on CSCS
        LOGGER.info("Clean uploaded files and results")
        working_dir.rmdir(HPCSchedulerClient.CSCS_DATA_FOLDER)

        LOGGER.info(encrypted_files)
        return encrypted_files

    @staticmethod
    def _handle_metric_results(metric_encrypted_file, metric_vm_encrypted_file, operation, encryption_handler):
        if not metric_encrypted_file:
            return None, None

        metric_op_dir, metric_op = BurstService.prepare_metrics_operation(operation)
        metric_files = encryption_handler.decrypt_files_to_dir([metric_encrypted_file,
                                                                metric_vm_encrypted_file], metric_op_dir)
        metric_file = metric_files[0]
        metric_vm = h5.load_view_model_from_file(metric_files[1])
        metric_op.view_model_gid = metric_vm.gid.hex
        dao.store_entity(metric_op)
        return metric_op, metric_file

    @staticmethod
    def stage_out_to_operation_folder(working_dir, operation, simulator_gid):
        # type: (Storage, Operation, typing.Union[uuid.UUID, str]) -> (list, Operation, str)
        encrypted_files = HPCSchedulerClient._stage_out_results(working_dir, simulator_gid)

        simulation_results = list()
        metric_encrypted_file = None
        metric_vm_encrypted_file = None
        for encrypted_file in encrypted_files:
            if os.path.basename(encrypted_file).startswith(DatatypeMeasureH5.file_name_base()):
                metric_encrypted_file = encrypted_file
            elif os.path.basename(encrypted_file).startswith(MEASURE_METRICS_MODEL_CLASS):
                metric_vm_encrypted_file = encrypted_file
            else:
                simulation_results.append(encrypted_file)

        encryption_handler = StorageInterface.get_encryption_handler(simulator_gid)
        metric_op, metric_file = HPCSchedulerClient._handle_metric_results(metric_encrypted_file,
                                                                           metric_vm_encrypted_file, operation,
                                                                           encryption_handler)
        project = dao.get_project_by_id(operation.fk_launched_in)
        operation_dir = HPCSchedulerClient.storage_interface.get_project_folder(project.name, str(operation.id))
        h5_filenames = encryption_handler.decrypt_files_to_dir(simulation_results, operation_dir)
        encryption_handler.cleanup_encryption_handler()
        LOGGER.info("Decrypted h5: {}".format(h5_filenames))
        LOGGER.info("Metric op: {}".format(metric_op))
        LOGGER.info("Metric file: {}".format(metric_file))

        return h5_filenames, metric_op, metric_file

    @staticmethod
    def _run_hpc_job(operation_identifier):
        # type: (int) -> None
        operation = dao.get_operation_by_id(operation_identifier)
        project_folder = HPCSchedulerClient.storage_interface.get_project_folder(operation.project.name)
        storage_interface = StorageInterface()
        storage_interface.inc_running_op_count(project_folder)
        is_group_launch = operation.fk_operation_group is not None
        simulator_gid = operation.view_model_gid
        try:
            HPCSchedulerClient._launch_job_with_pyunicore(operation, simulator_gid, is_group_launch)
        except Exception as exception:
            LOGGER.error("Failed to submit job HPC", exc_info=True)
            operation.mark_complete(STATUS_ERROR,
                                    exception.response.text if isinstance(exception, HTTPError) else repr(exception))
            dao.store_entity(operation)
        storage_interface.check_and_delete(project_folder)

    @staticmethod
    def _stage_out_outputs(encrypted_dir_path, output_list):
        # type: (str, dict) -> list
        if not os.path.isdir(encrypted_dir_path):
            os.makedirs(encrypted_dir_path)

        encrypted_files = list()
        for output_filename, output_filepath in output_list.items():
            if type(output_filepath) is not unicore_client.PathFile:
                LOGGER.info("Object {} is not a file.")
                continue
            filename = os.path.join(encrypted_dir_path, os.path.basename(output_filename))
            output_filepath.download(filename)
            encrypted_files.append(filename)
        return encrypted_files

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
