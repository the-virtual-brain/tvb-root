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
import cgi
import json
import os
import sys

import requests
from requests import HTTPError
from tvb.adapters.simulator.hpc_simulator_adapter import HPCSimulatorAdapter
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config.init.datatypes_registry import populate_datatypes_registry
from tvb.core.entities.model.model_operation import STATUS_STARTED, STATUS_FINISHED, STATUS_ERROR
from tvb.core.services.backend_clients.hpc_scheduler_client import EncryptionHandler
from tvb.core.services.simulator_serializer import SimulatorSerializer

log = get_logger('tvb.core.operation_hpc_launcher')

UPDATE_STATUS_KEY = "NEW_STATUS"

if __name__ == '__main__':
    TvbProfile.set_profile(TvbProfile.WEB_PROFILE)
    TvbProfile.current.hpc.IS_HPC_RUN = True


def do_operation_launch(simulator_gid, available_disk_space, is_group_launch, base_url):
    try:
        log.info("Preparing HPC launch for simulation with id={}".format(simulator_gid))
        populate_datatypes_registry()
        log.info("Current TVB profile has HPC run=: {}".format(TvbProfile.current.hpc.IS_HPC_RUN))
        encyrption_handler = EncryptionHandler(simulator_gid)
        _request_passfile(simulator_gid, base_url, os.path.dirname(encyrption_handler.get_passfile()))
        # TODO: Ensure encrypted_dir is correctly configured for CSCS
        plain_input_dir = encyrption_handler.open_plain_dir()
        log.info("Current wdir is: {}".format(plain_input_dir))
        view_model = SimulatorSerializer().deserialize_simulator(simulator_gid, plain_input_dir)
        adapter_instance = HPCSimulatorAdapter(plain_input_dir, is_group_launch)
        _update_operation_status(STATUS_STARTED, simulator_gid, base_url)
        adapter_instance._prelaunch(None, None, available_disk_space, view_model)
        _update_operation_status(STATUS_FINISHED, simulator_gid, base_url)

    except Exception as excep:
        log.error("Could not execute operation {}".format(str(sys.argv[1])))
        log.exception(excep)
        _update_operation_status(STATUS_ERROR, simulator_gid, base_url)

    finally:
        encyrption_handler.close_plain_dir()


# TODO: extract common rest api parts
CHUNK_SIZE = 128


def _save_file(file_path, response):
    with open(file_path, 'wb') as local_file:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                local_file.write(chunk)
    return file_path


def _request_passfile(simulator_gid, base_url, passfile_folder):
    # type: (str, str, str) -> str
    try:
        response = requests.get("{}/flow/encryption_config/{}".format(base_url, simulator_gid))
        if response.ok:
            content_disposition = response.headers['Content-Disposition']
            value, params = cgi.parse_header(content_disposition)
            file_name = params['filename']
            file_path = os.path.join(passfile_folder, os.path.basename(file_name))
            return _save_file(file_path, response)
    except HTTPError:
        log.warning(
            "Failed to request passfile from TVB server {} for simulator {}".format(base_url, simulator_gid))


def _update_operation_status(status, simulator_gid, base_url):
    # type: (str, str, str) -> None
    try:
        requests.put("{}/flow/update_status/{}".format(base_url, simulator_gid), json={
            UPDATE_STATUS_KEY: status,
        })
    except HTTPError:
        log.warning(
            "Failed to notify TVB server {} for simulator {} status update {}".format(base_url, simulator_gid, status))


if __name__ == '__main__':
    simulator_gid = sys.argv[1]
    available_disk_space = sys.argv[2]
    is_group_launch = json.loads(sys.argv[3].lower())
    base_url = sys.argv[4]

    do_operation_launch(simulator_gid, available_disk_space, is_group_launch, base_url)
