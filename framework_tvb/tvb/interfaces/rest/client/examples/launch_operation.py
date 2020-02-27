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

from tvb.adapters.datatypes.h5.region_mapping_h5 import RegionMappingH5
from tvb.adapters.uploaders.csv_connectivity_importer import CSVConnectivityImporterModel, CSVConnectivityImporter
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporterModel, RegionMappingImporter
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel, ZIPConnectivityImporter
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporterModel, ZIPSurfaceImporter
from tvb.basic.logger.builder import get_logger
from tvb.interfaces.rest.client.examples.utils import compute_tvb_data_path, monitor_operation
from tvb.interfaces.rest.client.tvb_client import TVBClient

if __name__ == '__main__':
    logger = get_logger(__name__)

    logger.info("Preparing client...")
    tvb_client = TVBClient("http://localhost:9090")

    logger.info("Requesting a list of users...")
    tvb_client.login("tvb_user", "pass")

    logger.info("Requesting projects for logged user")
    projects_of_user = tvb_client.get_project_list()
    assert len(projects_of_user) > 0
    logger.info("TVB has {} projects for this user".format(len(projects_of_user)))

    project_gid = projects_of_user[0].gid

    logger.info("Launching connectivity uploading operation...")
    connectivity_view_model = ZIPConnectivityImporterModel()
    connectivity_view_model.uploaded = compute_tvb_data_path('connectivity', 'connectivity_96.zip')
    connectivity_view_model.normalization = 'region'
    operation_gid = tvb_client.launch_operation(project_gid, ZIPConnectivityImporter, connectivity_view_model)
    monitor_operation(tvb_client, operation_gid)

    logger.info("Requesting the result of the connectivity uploading...")
    connectivity_result = tvb_client.get_operation_results(operation_gid)[0]

    logger.info("Launching surface uploading operation...")
    surface_view_model = ZIPSurfaceImporterModel()
    surface_view_model.uploaded = compute_tvb_data_path('surfaceData', 'cortex_16384.zip')
    surface_view_model.surface_type = "Cortical Surface"
    surface_view_model.should_center = False
    operation_gid = tvb_client.launch_operation(project_gid, ZIPSurfaceImporter, surface_view_model)
    monitor_operation(tvb_client, operation_gid)

    logger.info("Requesting the result of the surface uploading...")
    surface_result = tvb_client.get_operation_results(operation_gid)[0]

    logger.info("Downloading the connectivity from server...")
    connectivity_path = tvb_client.retrieve_datatype(connectivity_result.gid, tvb_client.temp_folder)

    logger.info("Downloading the surface from server...")
    surface_path = tvb_client.retrieve_datatype(surface_result.gid, tvb_client.temp_folder)

    logger.info("Launching region mapping upload operation...")
    rm_view_model = RegionMappingImporterModel()
    rm_view_model.mapping_file = compute_tvb_data_path('regionMapping', 'regionMapping_16k_76.txt')
    rm_view_model.connectivity = UUID(connectivity_result.gid)
    rm_view_model.surface = UUID(surface_result.gid)
    operation_gid = tvb_client.launch_operation(project_gid, RegionMappingImporter, rm_view_model)
    monitor_operation(tvb_client, operation_gid)

    logger.info("Downloading the region mapping uploaded above")
    region_mapping_gid = tvb_client.get_operation_results(operation_gid)[0].gid
    region_mapping_file_path = tvb_client.retrieve_datatype(region_mapping_gid, tvb_client.temp_folder)
    with RegionMappingH5(region_mapping_file_path) as rm:
        region_mapping_connectivity_gid = rm.connectivity.load().hex
    logger.info("Region mapping with gid {} is linked to a connectivity with gid {}".format(region_mapping_gid,
                                                                                            region_mapping_connectivity_gid))

    logger.info("Launching connectivity csv upload operation. Takes two files as input")
    csv_view_model = CSVConnectivityImporterModel()
    csv_view_model.weights = compute_tvb_data_path('dti_pipeline_toronto', 'output_ConnectionCapacityMatrix.csv')
    csv_view_model.tracts = compute_tvb_data_path('dti_pipeline_toronto', 'output_ConnectionDistanceMatrix.csv')
    csv = connectivity_result.gid
    csv_view_model.input_data = UUID(csv)
    operation_gid = tvb_client.launch_operation(project_gid, CSVConnectivityImporter, csv_view_model)
    monitor_operation(tvb_client, operation_gid)
