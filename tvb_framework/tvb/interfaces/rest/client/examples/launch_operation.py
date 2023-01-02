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
Example of using TVB importers from within the REST client API.
Here, we upload two independent datatypes: a connectivity and a surface from ZIP formats.
Then, we upload a region mapping that depends on both connectivity and surface to exist in TVB storage.
"""

from tvb.adapters.analyzers.bct_adapters import BaseBCTModel
from tvb.adapters.analyzers.bct_degree_adapters import Degree
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporterModel, RegionMappingImporter
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel, ZIPConnectivityImporter
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporterModel, ZIPSurfaceImporter
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.interfaces.rest.client.examples.utils import compute_tvb_data_path, monitor_operation, compute_rest_url
from tvb.interfaces.rest.client.tvb_client import TVBClient

logger = get_logger(__name__)


def launch_operation_examples(tvb_client_instance):
    logger.info("Requesting projects for logged user")
    projects_of_user = tvb_client_instance.get_project_list()
    assert len(projects_of_user) > 0
    logger.info("TVB has {} projects for this user".format(len(projects_of_user)))

    project_gid = projects_of_user[0].gid

    # --- Launch operations to import a Connectivity, a Surface and a RegionMapping ---

    logger.info("Importing a connectivity from ZIP...")
    zip_connectivity_importer_model = ZIPConnectivityImporterModel()
    zip_connectivity_importer_model.uploaded = compute_tvb_data_path('connectivity', 'connectivity_96.zip')
    zip_connectivity_importer_model.normalization = 'region'
    operation_gid = tvb_client_instance.launch_operation(project_gid, ZIPConnectivityImporter,
                                                         zip_connectivity_importer_model)
    monitor_operation(tvb_client_instance, operation_gid)

    logger.info("Get the result of connectivity import...")
    connectivity_dto = tvb_client_instance.get_operation_results(operation_gid)[0]

    logger.info("Importing a surface from ZIP...")
    zip_surface_importer_model = ZIPSurfaceImporterModel()
    zip_surface_importer_model.uploaded = compute_tvb_data_path('surfaceData', 'cortex_16384.zip')
    zip_surface_importer_model.surface_type = SurfaceTypesEnum.CORTICAL_SURFACE
    zip_surface_importer_model.should_center = False
    operation_gid = tvb_client_instance.launch_operation(project_gid, ZIPSurfaceImporter, zip_surface_importer_model)
    monitor_operation(tvb_client_instance, operation_gid)

    logger.info("Get the result of surface import...")
    surface_dto = tvb_client_instance.get_operation_results(operation_gid)[0]

    logger.info("Importing a region mapping...")
    rm_importer_model = RegionMappingImporterModel()
    rm_importer_model.mapping_file = compute_tvb_data_path('regionMapping', 'regionMapping_16k_76.txt')
    rm_importer_model.connectivity = connectivity_dto.gid
    rm_importer_model.surface = surface_dto.gid
    operation_gid = tvb_client_instance.launch_operation(project_gid, RegionMappingImporter, rm_importer_model)
    monitor_operation(tvb_client_instance, operation_gid)

    logger.info("Get the result of region mapping import...")
    region_mapping_dto = tvb_client_instance.get_operation_results(operation_gid)[0]

    # --- Load the region mapping together with references information in 3 different ways ---

    logger.info("1.Download and load the region mapping with all its references...")
    region_mapping_complete = tvb_client_instance.load_datatype_with_full_references(region_mapping_dto.gid,
                                                                                     tvb_client_instance.temp_folder)
    logger.info("1.This region mapping is linked to a connectivity with GID={} and number_of_regions={}".format(
        region_mapping_complete.connectivity.gid, region_mapping_complete.connectivity.number_of_regions))

    logger.info("2.Download and load the region mapping with only GIDs for its references...")
    region_mapping_with_links = tvb_client_instance.load_datatype_with_links(region_mapping_dto.gid,
                                                                             tvb_client_instance.temp_folder)
    logger.info("2.This region mapping is linked to a connectivity with GID={}".format(
        region_mapping_with_links.connectivity.gid))

    logger.info("3.Only download the region mapping on client machine...")
    region_mapping_path = tvb_client_instance.retrieve_datatype(region_mapping_dto.gid, tvb_client_instance.temp_folder)

    logger.info("3.Load the region mapping that was already downloaded on client machine...")
    local_region_mapping_with_links = tvb_client_instance.load_datatype_from_file(region_mapping_path)
    logger.info("3.This region mapping is linked to a connectivity with GID={}".format(
        local_region_mapping_with_links.connectivity.gid))

    # --- Launch operation to run a Degree Analyzer over the Connectivity ---

    bct_model = BaseBCTModel()
    bct_model.connectivity = connectivity_dto.gid
    operation_gid = tvb_client_instance.launch_operation(project_gid, Degree, bct_model)
    monitor_operation(tvb_client_instance, operation_gid)

    logger.info("Get the result of BCT...")
    bct_dto = tvb_client_instance.get_operation_results(operation_gid)[0]
    logger.info("The resulted BCT has GID={}".format(bct_dto.gid))


if __name__ == '__main__':
    logger.info("Preparing client...")
    tvb_client = TVBClient(compute_rest_url())

    logger.info("Attempt to login")
    tvb_client.browser_login()
    launch_operation_examples(tvb_client)
