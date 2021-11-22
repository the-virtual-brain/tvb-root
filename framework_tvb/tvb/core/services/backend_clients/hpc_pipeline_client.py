# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import os
from datetime import datetime

from requests import HTTPError
from tvb.adapters.creators.pipeline_creator import IPPipelineCreatorModel
from tvb.basic.config.settings import HPCSettings
from tvb.basic.exceptions import TVBException
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.model.model_datatype import ZipDatatype
from tvb.core.entities.model.model_operation import STATUS_ERROR
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.backend_clients.hpc_client import HPCClient, HPCOperationThread
from tvb.core.services.exceptions import OperationException
from tvb.storage.storage_interface import StorageInterface

try:
    import pyunicore.client as unicore_client
    from pyunicore.client import Job, Storage, Client
except ImportError:
    HPCSettings.CAN_RUN_HPC = False

LOGGER = get_logger(__name__)

HPC_THREADS = []


class HPCPipelineClient(HPCClient):
    """
    HPC backend client to run the image processing pipeline.
    """
    SCRIPT_FOLDER_NAME = "ebrains"
    PUBLIC_KEY_FILENAME = "public_key_pipeline_keys.pem"
    INPUT_FILES_CSCS_FOLDER = 'input'
    RESULTS_KEYS_FOLDER = "results_keys"

    @staticmethod
    def _prepare_input(operation):
        """
        Gather raw images to process with pipeline on HPC
        :param operation:
        :return: list of input files [input dataset zip, arguments file]
        """
        storage_path = StorageInterface().get_project_folder(operation.project.name,
                                                             str(operation.id))
        pipeline_data_zip = os.path.join(storage_path, IPPipelineCreatorModel.PIPELINE_DATASET_FILE)
        if not os.path.exists(pipeline_data_zip):
            raise TVBException(
                "Pipeline input data was not found in the operation folder for operation {}".format(operation.id))

        args_file = os.path.join(storage_path, IPPipelineCreatorModel.PIPELINE_CONFIG_FILE)
        if not os.path.exists(args_file):
            raise TVBException(
                "Arguments file for the pipeline script was not found in the operation folder for operation {}".format(
                    operation.id))
        return [pipeline_data_zip, args_file]

    @staticmethod
    def _configure_job(operation_id, mode):
        # type: (int, int) -> (dict, str)
        bash_entrypoint = os.path.join(os.environ[HPCClient.TVB_BIN_ENV_KEY], HPCPipelineClient.SCRIPT_FOLDER_NAME,
                                       HPCSettings.HPC_PIPELINE_LAUNCHER_SH_SCRIPT)

        # Build job configuration JSON
        # TODO: correct parameters for pipeline to be added (mode, args for containers etc)
        my_job = {HPCSettings.UNICORE_JOB_NAME: 'PipelineProcessing_{}'.format(operation_id),
                  HPCSettings.UNICORE_RESOURCER_KEY: {'Runtime': '23h'},
                  HPCSettings.UNICORE_EXE_KEY: 'sh ' + os.path.basename(bash_entrypoint),
                  HPCSettings.UNICORE_ARGS_KEY: ['-m {}'.format(mode), '-p $PWD']}

        # TODO: Maybe take HPC Project also from GUI?
        if HPCClient.CSCS_PROJECT in os.environ:
            my_job[HPCSettings.UNICORE_PROJECT_KEY] = os.environ[HPCClient.CSCS_PROJECT]

        return my_job, bash_entrypoint

    @staticmethod
    def _launch_job_with_pyunicore(operation, authorization_token):
        # type: (Operation, str) -> Job
        LOGGER.info("Prepare job configuration for operation: {}".format(operation.id))
        job_config, job_script = HPCPipelineClient._configure_job(operation.id, 4)

        LOGGER.info("Prepare input files: pipeline input data zip file and pipeline arguments json file. ")
        inputs = HPCPipelineClient._prepare_input(operation)

        LOGGER.info("Prepare unicore job")
        job = HPCClient._prepare_pyunicore_job(operation=operation, job_inputs=inputs, job_script=job_script,
                                               job_config=job_config,
                                               auth_token=authorization_token, inputs_subfolder=None)
        mount_point = job.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]

        site_client = HPCClient._build_unicore_client(authorization_token,
                                                      unicore_client._HBP_REGISTRY_URL,
                                                      TvbProfile.current.hpc.HPC_COMPUTE_SITE)

        # TODO: Should we run these steps or these will be run from sh script?
        zip_path = os.path.join(mount_point, IPPipelineCreatorModel.PIPELINE_DATASET_FILE)
        unzip_path = os.path.join(mount_point, 'Demo_data_pipeline_CON03')
        unzip_archive = site_client.execute('unzip {} -d {}'.format(zip_path, unzip_path))
        unzip_mount_point = unzip_archive.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]
        LOGGER.info("Unzip mount point: {}".format(unzip_mount_point))
        HPCPipelineClient._poll_job(unzip_archive)

        script_path = os.path.join(mount_point, HPCSettings.HPC_PIPELINE_LAUNCHER_SH_SCRIPT)
        install_datalad = site_client.execute('sh {} -m 0 -p {}'.format(script_path, os.path.normpath(mount_point)))
        install_datalad_mount_point = install_datalad.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]
        LOGGER.info("Insall datalad mount point: {}".format(install_datalad_mount_point))
        HPCPipelineClient._poll_job(install_datalad)
        HPCPipelineClient._transfer_logs(site_client, install_datalad, job, 'install_datalad')

        pull_fmri = site_client.execute('sh {} -m 11 -p {}'.format(script_path, os.path.normpath(mount_point)))
        pull_fmri_mount_point = pull_fmri.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]
        LOGGER.info("Pull fmri mount point: {}".format(pull_fmri_mount_point))
        HPCPipelineClient._poll_job(pull_fmri)
        HPCPipelineClient._transfer_logs(site_client, pull_fmri, job, 'pull_fmri')

        create_datasets = site_client.execute('sh {} -m 2 -p {}'.format(script_path, os.path.normpath(mount_point)))
        create_datasets_mount_point = create_datasets.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]
        LOGGER.info("Pull fmri mount point: {}".format(create_datasets_mount_point))
        HPCPipelineClient._poll_job(create_datasets)
        HPCPipelineClient._transfer_logs(site_client, create_datasets, job, 'create_datasets')

        try:
            LOGGER.info("Start unicore job")
            job.start()
            LOGGER.info("Job has started")
        except Exception as e:
            LOGGER.error('Cannot start unicore job for operation {}'.format(operation.id), e)
            raise TVBException(e)
        return job

    @staticmethod
    def _transfer_logs(site_client, from_job, to_job, files_prefix, to_folder='intermediary_logs'):
        from_job_mount_point = from_job.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]
        to_job_mount_point = to_job.working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]
        LOGGER.info("Transfer logs from {} to {}".format(from_job_mount_point, to_job_mount_point))
        make_logs_dir = site_client.execute("mkdir -p {}{}".format(to_job_mount_point, to_folder))
        make_logs_dir.poll()
        command = 'cp {}{} {}{}/{}-{}'
        stderr_command = command.format(from_job_mount_point, 'stderr', to_job_mount_point, to_folder, files_prefix,
                                        'stderr')
        stdout_command = command.format(from_job_mount_point, 'stdout', to_job_mount_point, to_folder, files_prefix,
                                        'stdout')
        transfer_logs = site_client.execute('{} && {}'.format(stderr_command, stdout_command))
        transfer_logs.poll()

    @staticmethod
    def _generate_results_keys(storage_path):
        enc_handler = StorageInterface.get_import_export_encryption_handler()
        results_key_folder = os.path.join(storage_path, HPCPipelineClient.RESULTS_KEYS_FOLDER)
        StorageInterface().check_created(results_key_folder)
        enc_handler.generate_public_private_key_pair(results_key_folder)
        results_public_key = os.path.join(results_key_folder, enc_handler.PUBLIC_KEY_NAME)
        return results_public_key

    @staticmethod
    def _generate_public_private_keys_hpc(mount_point):
        site_client = HPCClient._build_unicore_client(os.environ[HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY],
                                                      unicore_client._HBP_REGISTRY_URL,
                                                      TvbProfile.current.hpc.HPC_COMPUTE_SITE)
        script_path = os.path.join(mount_point, HPCSettings.HPC_PIPELINE_LAUNCHER_SH_SCRIPT)
        mkdir_job = site_client.execute('mkdir {}'.format(os.path.join(mount_point, 'keys')))
        mkdir_job.poll()
        generate_keys_job = site_client.execute('sh {} -m 5 -p {}'.format(script_path, os.path.normpath(mount_point)))
        generate_keys_job.poll()

    @staticmethod
    def _run_hpc_job(operation_identifier, authorization_token):
        # type: (int, str) -> None
        operation = dao.get_operation_by_id(operation_identifier)
        project_folder = HPCClient.storage_interface.get_project_folder(operation.project.name)
        storage_interface = StorageInterface()
        storage_interface.inc_running_op_count(project_folder)

        try:
            HPCPipelineClient._launch_job_with_pyunicore(operation, authorization_token)
        except Exception as exception:
            LOGGER.error("Failed to submit job HPC", exc_info=True)
            operation.mark_complete(STATUS_ERROR,
                                    exception.response.text if isinstance(exception, HTTPError) else repr(exception))
            dao.store_entity(operation)
        storage_interface.check_and_delete(project_folder)

    @staticmethod
    def execute(operation_id, user_name_label, adapter_instance, auth_token=""):
        # type: (int, None, None) -> None
        """
        Submit an operation asynchronously on HPC
        """
        thread = HPCOperationThread(operation_id, target=HPCPipelineClient._run_hpc_job,
                                    kwargs={'operation_identifier': operation_id, 'authorization_token': auth_token})
        thread.start()
        HPC_THREADS.append(thread)

    @staticmethod
    def stop_operation(operation_id):
        """
        Stop the thread for a given operation id
        """
        return True

    @staticmethod
    def _stage_out_results(working_dir):
        # type: (Storage) -> list
        output_list = HPCClient._listdir(working_dir)
        LOGGER.info("Output list {}".format(output_list))
        storage_interface = StorageInterface()
        # TODO: Choose a local folder to copy back the encrypted results
        now = datetime.now()
        date_str = "%d-%d-%d_%d-%d-%d_%d" % (now.year, now.month, now.day, now.hour,
                                             now.minute, now.second, now.microsecond)
        uq_name = "PipelineResults-%s" % date_str
        unique_Path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, uq_name)
        encrypted_files = HPCClient._stage_out_outputs(unique_Path, output_list)

        # Clean data uploaded on CSCS
        LOGGER.info("Clean uploaded files and results")
        # working_dir.rmdir(HPCClient.CSCS_DATA_FOLDER)

        LOGGER.info(encrypted_files)
        return encrypted_files

    @staticmethod
    def stage_out_logs(working_dir, operation):
        # type: (Storage, Operation) -> str
        now = datetime.now()
        date_str = "%d-%d-%d_%d-%d-%d_%d" % (now.year, now.month, now.day, now.hour,
                                             now.minute, now.second, now.microsecond)
        uq_name = "PipelineResults-%s" % date_str
        local_logs_dir = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, uq_name)

        if not os.path.isdir(local_logs_dir):
            os.makedirs(local_logs_dir)

        stderr_f = 'stderr'
        stdout_f = 'stdout'
        stderr = working_dir.stat(stderr_f)
        stdout = working_dir.stat(stdout_f)

        stderr.download(os.path.join(local_logs_dir, stderr_f))
        stdout.download(os.path.join(local_logs_dir, stdout_f))

        try:
            output_list = HPCClient._listdir(working_dir, base='intermediary_logs')

            for output_filename, output_filepath in output_list.items():
                if type(output_filepath) is not unicore_client.PathFile:
                    LOGGER.info("Object {} is not a file.")
                    continue
                filename = os.path.join(local_logs_dir, os.path.basename(output_filename))
                output_filepath.download(filename)

        except OperationException as e:
            LOGGER.warning("Exception when trying to download intermediary logs from HPC: {}".format(e.message))

        project = dao.get_project_by_id(operation.fk_launched_in)
        operation_dir = HPCClient.storage_interface.get_project_folder(project.name, str(operation.id))
        results_zip = os.path.join(operation_dir, 'pipeline_results.zip')
        StorageInterface().write_zip_folder(results_zip, local_logs_dir)
        # StorageInterface.remove_folder(local_logs_dir)

        return results_zip

    @staticmethod
    def stage_out_to_operation_folder(working_dir, operation):
        # type: (Storage, Operation) -> (list, Operation, str)
        encrypted_files = HPCPipelineClient._stage_out_results(working_dir)

        project = dao.get_project_by_id(operation.fk_launched_in)
        operation_dir = HPCClient.storage_interface.get_project_folder(project.name, str(operation.id))

        # TODO: decrypt pipeline results under the operation dir?
        assert len(encrypted_files) > 0
        tmp_folder = os.path.basename(encrypted_files[0])
        results_zip = os.path.join(operation_dir, 'pipeline_results.zip')
        StorageInterface().write_zip_folder(results_zip, tmp_folder)
        StorageInterface.remove_folder(tmp_folder)

        return results_zip

    @staticmethod
    def update_db_with_results(operation, results_zip):
        # type: (Operation, str) -> (str, int)
        project = dao.get_project_by_id(operation.fk_launched_in)
        op_dir = HPCClient.storage_interface.get_project_folder(project.name, str(operation.id))
        view_model = h5.load_view_model(operation.view_model_gid, op_dir)

        ga = GenericAttributes()
        ga.fill_from(view_model.generic_attributes)

        index = ZipDatatype()
        index.fill_from_generic_attributes(ga)
        index.fk_from_operation = operation.id
        index.zip_path = results_zip
        dao.store_entity(index)
