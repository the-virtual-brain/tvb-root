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
Import connectivities from Brain Tumor zip archives

.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""
import sys

from tvb.basic.logger.builder import get_logger
from tvb.config import DEFAULT_PROJECT_GID
from tvb.interfaces.command.lab import *

CONN_ZIP_FILE = "SC.zip"
LOG = get_logger(__name__)


def import_tumor_connectivities(project_id, folder_path):
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
                data = {
                    'data_subject': patient,
                    'generic_attributes.user_tag_1': user_tag
                }
                import_op = import_conn_zip(project_id, connectivity_zip, data)
                # wait_to_finish(import_op)


if __name__ == '__main__':
    default_prj_id = dao.get_project_by_gid(DEFAULT_PROJECT_GID).id

    # Path to TVB folder
    input_folder = sys.argv[1]
    import_tumor_connectivities(default_prj_id, input_folder)
