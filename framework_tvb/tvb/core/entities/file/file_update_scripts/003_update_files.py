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
Upgrade script from H5 version 2 to version 3

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. creationdate:: beginning of 2015
"""

import os
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.exceptions import FileVersioningException
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.import_service import ImportService


def update_localconnectivity_metadata(folder, file_name):
    service = ImportService()
    operation_id = int(os.path.split(folder)[1])

    dt = service.load_datatype_from_file(os.path.join(folder, file_name), operation_id)
    info_dict = {"dtype": dt.matrix.dtype.str,
                 "format": dt.matrix.format,
                 "Shape": str(dt.matrix.shape),
                 "Maximum": dt.matrix.data.max(),
                 "Minimum": dt.matrix.data.min(),
                 "Mean": dt.matrix.mean()}
    dt.set_metadata(info_dict, '', True, '/matrix')


def update(input_file, burst_match_dict=None):
    """
    In order to avoid segmentation faults when updating a batch of files just
    start every conversion on a different Python process.

    :param input_file: the file that needs to be converted to a newer file storage version.
        This should be a file that still uses TVB 2.0 storage
    """
    if not os.path.isfile(input_file):
        raise FileVersioningException("The input path %s received for upgrading from 2 -> 3 is not a "
                                      "valid file on the disk." % input_file)

    folder, file_name = os.path.split(input_file)
    storage_manager = HDF5StorageManager(folder, file_name)

    root_metadata = storage_manager.get_metadata()
    class_name = root_metadata[DataTypeMetaData.KEY_CLASS_NAME]

    if class_name == "LocalConnectivity":
        root_metadata[DataTypeMetaData.KEY_MODULE] = "tvb.datatypes.local_connectivity"
        storage_manager.set_metadata(root_metadata)
        update_localconnectivity_metadata(folder, file_name)

    elif class_name == "RegionMapping":
        root_metadata[DataTypeMetaData.KEY_MODULE] = "tvb.datatypes.region_mapping"

    root_metadata[TvbProfile.current.version.DATA_VERSION_ATTRIBUTE] = TvbProfile.current.version.DATA_VERSION
    storage_manager.set_metadata(root_metadata)
