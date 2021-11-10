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

from requests import HTTPError
from tvb.basic.config.settings import HPCSettings
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_operation import STATUS_ERROR
from tvb.core.entities.storage import dao
from tvb.core.services.backend_clients.hpc_client import HPCClient
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
        LOGGER.info("Prepare job inputs for operation: {}".format(operation.id))
        job_plain_inputs = HPCPipelineClient._prepare_input(operation)

        LOGGER.info("Prepare job configuration for operation: {}".format(operation.id))
        job_config, job_script = HPCPipelineClient._configure_job(operation.id)

        LOGGER.info("Encrypt job inputs for operation: {}".format(operation.id))
        # TODO: encrypt inputs before stage-in!!!
        job_encrypted_inputs = job_plain_inputs

        job = HPCClient._prepare_pyunicore_job(operation, job_encrypted_inputs, job_script, job_config)
        return job

    @staticmethod
    def _run_hpc_job(operation_id):
        # type: (int) -> None
        operation = dao.get_operation_by_id(operation_id)
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
    def stop_operation(operation_id):
        """
        Stop the thread for a given operation id
        """
        return True
