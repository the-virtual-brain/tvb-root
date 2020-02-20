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

from uuid import UUID
import time
import tvb_data
import os
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel, ZIPConnectivityImporter
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporterModel, ZIPSurfaceImporter
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporterModel, RegionMappingImporter
from tvb.adapters.uploaders.csv_connectivity_importer import CSVConnectivityImporterModel, CSVConnectivityImporter
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model.model_operation import STATUS_ERROR, STATUS_CANCELED, STATUS_FINISHED
from tvb.core.entities.storage import dao
from tvb.interfaces.rest.client.tvb_client import TVBClient

if __name__ == '__main__':

    logger = get_logger(__name__)

    logger.info("Preparing client...")
    tvb_client = TVBClient("http://localhost:9090")

    logger.info("Requesting a list of users...")
    users_list = tvb_client.get_users()
    assert len(users_list) > 0
    logger.info("TVB has {} users registered".format(len(users_list)))

    username = users_list[0].username
    logger.info("Requesting projects for user {}...".format(username))
    projects_of_user = tvb_client.get_project_list(username)
    assert len(projects_of_user) > 0
    logger.info("TVB has {} projects for this user".format(len(projects_of_user)))

    project_gid = projects_of_user[0].gid

    logger.info("Preparing a connectivity H5 file...")
    connectivity_view_model = ZIPConnectivityImporterModel()
    conn_zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_96.zip')
    connectivity_view_model.uploaded = conn_zip_path
    connectivity_view_model.normalization = 'region'

    logger.info("Launching connectivity uploading operation...")
    operation_gid = tvb_client.launch_operation(project_gid, ZIPConnectivityImporter, connectivity_view_model)

    while True:
        status = tvb_client.get_operation_status(operation_gid)
        if status in [STATUS_FINISHED, STATUS_CANCELED, STATUS_ERROR]:
            break
        time.sleep(5)
    logger.info("The connectivity uploading has finished with status: {}".format(status))

    logger.info("Requesting the result of the connectivity uploading...")
    connectivity_result = tvb_client.get_operation_results(operation_gid)[0]

    logger.info("Preparing a surface H5 file...")
    surface_view_model = ZIPSurfaceImporterModel()
    surface_zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'surfaceData', 'cortex_16384.zip')
    surface_view_model.uploaded = surface_zip_path
    surface_view_model.surface_type = "Cortical Surface"
    surface_view_model.should_center = False

    logger.info("Launching surface uploading operation...")
    operation_gid = tvb_client.launch_operation(project_gid, ZIPSurfaceImporter, surface_view_model)

    while True:
        status = tvb_client.get_operation_status(operation_gid)
        if status in [STATUS_FINISHED, STATUS_CANCELED, STATUS_ERROR]:
            break
        time.sleep(5)
    logger.info("The surface uploading has finished with status: {}".format(status))

    logger.info("Requesting the result of the surface uploading...")
    surface_result = tvb_client.get_operation_results(operation_gid)[0]

    logger.info("Downloading the connectivity from server...")
    connectivity_path = tvb_client.retrieve_datatype(connectivity_result.gid, TvbProfile.current.TVB_TEMP_FOLDER)

    logger.info("Downloading the surface from server...")
    surface_path = tvb_client.retrieve_datatype(surface_result.gid, TvbProfile.current.TVB_TEMP_FOLDER)

    logger.info("Preparing a region mapping H5 file using the downloaded connectivity and surface...")
    rm_view_model = RegionMappingImporterModel()
    rm_text_path = os.path.join(os.path.dirname(tvb_data.__file__), 'regionMapping', 'regionMapping_16k_76.txt')
    rm_view_model.mapping_file = rm_text_path
    conn_db = dao.get_datatype_by_gid(connectivity_result.gid).gid
    rm_view_model.connectivity = UUID(conn_db)
    surface_db = dao.get_datatype_by_gid(surface_result.gid).gid
    rm_view_model.surface = UUID(surface_db)

    logger.info("Launching region mapping upload operation...")
    operation_gid = tvb_client.launch_operation(project_gid, RegionMappingImporter, rm_view_model)

    while True:
        status = tvb_client.get_operation_status(operation_gid)
        if status in [STATUS_FINISHED, STATUS_CANCELED, STATUS_ERROR]:
            break
        time.sleep(5)
    logger.info("The region mapping uploading has finished with status: {}".format(status))
    os.remove(connectivity_path)
    os.remove(surface_path)

    logger.info("Preparing a connectivity csv H5 file (this one requires two datatype files: one containing weights and the other one containing tracts)...")
    csv_view_model = CSVConnectivityImporterModel()

    csv_weights_path = os.path.join(os.path.dirname(tvb_data.__file__), 'dti_pipeline_toronto', 'output_ConnectionCapacityMatrix.csv')
    csv_tracts_path = os.path.join(os.path.dirname(tvb_data.__file__), 'dti_pipeline_toronto', 'output_ConnectionDistanceMatrix.csv')
    csv_view_model.weights = csv_weights_path
    csv_view_model.tracts = csv_tracts_path
    csv = dao.get_datatype_by_gid(connectivity_result.gid).gid
    csv_view_model.input_data = UUID(csv)

    logger.info("Launching connectivity csv upload operation...")
    operation_gid = tvb_client.launch_operation(project_gid, CSVConnectivityImporter, csv_view_model)

    while True:
        status = tvb_client.get_operation_status(operation_gid)
        if status in [STATUS_FINISHED, STATUS_CANCELED, STATUS_ERROR]:
            break
        time.sleep(5)
    logger.info("The connectivity csv uploading has finished with status: {}".format(status))

