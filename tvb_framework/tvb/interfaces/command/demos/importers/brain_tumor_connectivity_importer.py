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
Import connectivities from Brain Tumor zip archives

.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""
import sys
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporter, RegionMappingImporterModel
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporter, ZIPSurfaceImporterModel
from tvb.basic.logger.builder import get_logger
from tvb.basic.readers import try_get_absolute_path
from tvb.interfaces.command.lab import *

CONN_ZIP_FILE = "SC.zip"
LOG = get_logger(__name__)


def import_tumor_connectivities(project_id, folder_path):
    conn_gids = []
    for patient in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient)
        if os.path.isdir(patient_path):
            user_tags = os.listdir(patient_path)
            for user_tag in user_tags:
                conn_folder = os.path.join(patient_path, user_tag)
                connectivity_zip = os.path.join(conn_folder, CONN_ZIP_FILE)
                if not os.path.exists(connectivity_zip):
                    LOG.error("File {} does not exist.".format(connectivity_zip))
                    continue
                import_conn_adapter = ABCAdapter.build_adapter_from_class(ZIPConnectivityImporter)
                import_conn_model = ZIPConnectivityImporterModel()
                import_conn_model.uploaded = connectivity_zip
                import_conn_model.data_subject = patient
                import_conn_model.generic_attributes.user_tag_1 = user_tag
                import_op = fire_operation(project_id, import_conn_adapter, import_conn_model)
                import_op = wait_to_finish(import_op)
                conn_gids.append(dao.get_results_for_operation(import_op.id)[0].gid)
    return conn_gids


def import_surface_rm(project_id, conn_gid):
    # Import surface and region mapping from tvb_data berlin subjects (68 regions)
    rm_file = try_get_absolute_path("tvb_data", "berlinSubjects/DH_20120806/DH_20120806_RegionMapping.txt")
    surface_zip_file = try_get_absolute_path("tvb_data", "berlinSubjects/DH_20120806/DH_20120806_Surface_Cortex.zip")

    surface_importer = ABCAdapter.build_adapter_from_class(ZIPSurfaceImporter)
    surface_imp_model = ZIPSurfaceImporterModel()
    surface_imp_model.uploaded = surface_zip_file
    surface_imp_operation = fire_operation(project_id, surface_importer, surface_imp_model)
    surface_imp_operation = wait_to_finish(surface_imp_operation)

    surface_gid = dao.get_results_for_operation(surface_imp_operation.id)[0].gid
    rm_importer = ABCAdapter.build_adapter_from_class(RegionMappingImporter)
    rm_imp_model = RegionMappingImporterModel()
    rm_imp_model.mapping_file = rm_file
    rm_imp_model.surface = surface_gid
    rm_imp_model.connectivity = conn_gid
    rm_import_operation = fire_operation(project_id, rm_importer, rm_imp_model)
    wait_to_finish(rm_import_operation)


if __name__ == '__main__':
    # Path to TVB folder
    input_folder = sys.argv[1]
    # Project where DTs will be imported
    project_id = sys.argv[2]

    conn_gids = import_tumor_connectivities(project_id, input_folder)
    import_surface_rm(project_id, conn_gids[0])
