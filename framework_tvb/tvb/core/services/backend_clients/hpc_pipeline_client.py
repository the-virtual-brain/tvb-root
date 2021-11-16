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
import shutil

from requests import HTTPError
from tvb.basic.config.settings import HPCSettings
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_datatype import ZipDatatype
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model.model_operation import STATUS_ERROR
from tvb.core.entities.storage import dao
from tvb.core.services.backend_clients.hpc_client import HPCClient, HPCOperationThread
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
        :return:
        """
        # TODO: return correct input data here
        return []

    @staticmethod
    def _configure_job(operation_id):
        # type: (int, int) -> (dict, str)
        bash_entrypoint = os.path.join(os.environ[HPCClient.TVB_BIN_ENV_KEY], HPCPipelineClient.SCRIPT_FOLDER_NAME,
                                       HPCSettings.HPC_PIPELINE_LAUNCHER_SH_SCRIPT)

        # Build job configuration JSON
        # TODO: correct parameters for pipeline to be added (mode, args for containers etc)
        my_job = {HPCSettings.UNICORE_EXE_KEY: os.path.basename(bash_entrypoint),
                  HPCSettings.UNICORE_ARGS_KEY: [HPCClient.HOME_FOLDER_MOUNT,
                                                 operation_id],
                  HPCSettings.UNICORE_RESOURCER_KEY: {"CPUs": "1"}}

        if HPCClient.CSCS_PROJECT in os.environ:
            my_job[HPCSettings.UNICORE_PROJECT_KEY] = os.environ[HPCClient.CSCS_PROJECT]

        return my_job, bash_entrypoint

    @staticmethod
    def _launch_job_with_pyunicore(operation):
        # type: (Operation) -> Job
        # Step1 prepare main job
        LOGGER.info("Prepare job configuration for operation: {}".format(operation.id))
        job_config, job_script = HPCPipelineClient._configure_job(operation.id)
        job = HPCClient._prepare_pyunicore_job(operation, [], job_script, job_config)
        job_working_dir = job.working_dir
        mount_point = job_working_dir.properties[HPCSettings.JOB_MOUNT_POINT_KEY]

        script_path = os.path.join(mount_point, HPCSettings.HPC_PIPELINE_LAUNCHER_SH_SCRIPT)

        site_client = HPCClient._build_unicore_client(os.environ[HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY],
                                                      unicore_client._HBP_REGISTRY_URL,
                                                      TvbProfile.current.hpc.HPC_COMPUTE_SITE)
        install_datalad = site_client.execute('sh {} -m 0 -p {}'.format(script_path, os.path.normpath(mount_point)))
        install_datalad.poll()
        pull_containers_DataLad1 = site_client.execute(
            'sh {} -m 1 -p {}'.format(script_path, os.path.normpath(mount_point)))

        pull_containers_DataLad2 = site_client.execute(
            'sh {} -m 11 -p {}'.format(script_path, os.path.normpath(mount_point)))
        pull_containers_DataLad3 = site_client.execute(
            'sh {} -m 111 -p {}'.format(script_path, os.path.normpath(mount_point)))

        pull_containers_DataLad1.poll()
        pull_containers_DataLad2.poll()
        pull_containers_DataLad3.poll()

        HPCPipelineClient._upload_file_with_pyunicore(job_working_dir,
                                                      '/Users/bvalean/Downloads/Demo_data_pipeline_CON03.zip',
                                                      subfolder=None)

        zip_path = os.path.join(mount_point, 'Demo_data_pipeline_CON03.zip')
        unzip_archive = site_client.execute('unzip {} -d {}'.format(zip_path, mount_point))
        unzip_archive.poll()

        dataset_anal = site_client.execute('sh {} -m 2 -p {}'.format(script_path, os.path.normpath(mount_point)))
        dataset_anal.poll()

        run_job = site_client.execute('sh {} -m 4 -p {}'.format(script_path, os.path.normpath(mount_point)))
        run_job.poll()
        # Step2 generate PRIVATE-PUBLIC Keys pair on HPC

        # HPCPipelineClient._generate_public_private_keys_hpc(mount_point)

        # Step3 download public key from HPC in operation's folder
        # storage_path = StorageInterface().get_project_folder(operation.project.name,
        #                                                      str(operation.id))
        # input_files_public_key_path = os.path.join(storage_path, HPCPipelineClient.PUBLIC_KEY_FILENAME)
        # job_working_dir.stat(os.path.join('keys', HPCPipelineClient.PUBLIC_KEY_FILENAME)).download(
        #     input_files_public_key_path)

        # Step4 Prepare job input files. Encrypt these files using downloaded public key
        # LOGGER.info("Prepare job inputs for operation: {}".format(operation.id))
        # job_plain_inputs = HPCPipelineClient._prepare_input(operation)
        # LOGGER.info("Encrypt job inputs for operation: {}".format(operation.id))
        # TODO: encrypt inputs before stage-in!!!
        # job_encrypted_inputs = job_plain_inputs

        # Step5 Generate a PRIVATE-PUBLIC key pair for results encryption
        # results_public_key = HPCPipelineClient._generate_results_keys(storage_path)
        # job_encrypted_inputs.append(results_public_key)

        # Step6 Upload encrypted files and the public key used for results encryption
        # for input_file in job_encrypted_inputs:
        #     HPCPipelineClient._upload_file_with_pyunicore(job_working_dir, input_name=input_file,
        #                                                   subfolder=HPCPipelineClient.INPUT_FILES_CSCS_FOLDER)

        # Step7 Start pipeline job
        # job.start()
        return job

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
    def _run_hpc_job(operation_identifier):
        # type: (int) -> None
        operation = dao.get_operation_by_id(operation_identifier)
        project_folder = HPCClient.storage_interface.get_project_folder(operation.project.name)
        storage_interface = StorageInterface()
        storage_interface.inc_running_op_count(project_folder)

        try:
            HPCPipelineClient._launch_job_with_pyunicore(operation)
        except Exception as exception:
            LOGGER.error("Failed to submit job HPC", exc_info=True)
            operation.mark_complete(STATUS_ERROR,
                                    exception.response.text if isinstance(exception, HTTPError) else repr(exception))
            dao.store_entity(operation)
        storage_interface.check_and_delete(project_folder)

    @staticmethod
    def execute(operation_id, user_name_label, adapter_instance):
        # type: (int, None, None) -> None
        """
        Submit an operation asynchronously on HPC
        """
        thread = HPCOperationThread(operation_id, target=HPCPipelineClient._run_hpc_job,
                                    kwargs={'operation_identifier': operation_id})
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
        output_subfolder = HPCClient.CSCS_DATA_FOLDER  # + '/' + HPCClient.OUTPUT_FOLDER
        output_list = HPCClient._listdir(working_dir, output_subfolder)
        LOGGER.info("Output list {}".format(output_list))
        storage_interface = StorageInterface()
        # TODO: Choose a local folder to copy back the encrypted results
        encrypted_dir = os.path.join("enc_dir",
                                     HPCClient.OUTPUT_FOLDER)
        encrypted_files = HPCClient._stage_out_outputs(encrypted_dir, output_list)

        # Clean data uploaded on CSCS
        LOGGER.info("Clean uploaded files and results")
        # working_dir.rmdir(HPCClient.CSCS_DATA_FOLDER)

        LOGGER.info(encrypted_files)
        return encrypted_files

    @staticmethod
    def stage_out_to_operation_folder(working_dir, operation):
        # type: (Storage, Operation) -> (list, Operation, str)
        encrypted_files = HPCPipelineClient._stage_out_results(working_dir)

        project = dao.get_project_by_id(operation.fk_launched_in)
        operation_dir = HPCClient.storage_interface.get_project_folder(project.name, str(operation.id))

        # TODO: decrypt pipeline results under the operation dir?
        assert len(encrypted_files) == 1
        result_path = shutil.copy(encrypted_files[0], operation_dir)

        # TODO: unzip and try to import tvb-ready data
        # tvb_data_dir = os.path.join(results_dir, 'tvb-ready')

        return result_path

    @staticmethod
    def update_db_with_results(operation, results_zip):
        # type: (Operation, str) -> (str, int)
        index = ZipDatatype()
        index.fk_from_operation = operation.id
        index.zip_path = results_zip
        dao.store_entity(index)
