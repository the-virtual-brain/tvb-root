# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
"""

import os
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.exceptions import FileVersioningException
from tvb.core.services.import_service import ImportService
from tvb.core.traits.types_mapped import SparseMatrix, MappedType



def _update_localconnectivity_metadata(dt):

    mtx = dt.matrix
    info_dict = {SparseMatrix.DTYPE_META: mtx.dtype.str,
                 SparseMatrix.FORMAT_META: mtx.format,
                 MappedType.METADATA_ARRAY_SHAPE: str(mtx.shape),
                 MappedType.METADATA_ARRAY_MAX: mtx.data.max(),
                 MappedType.METADATA_ARRAY_MIN: mtx.data.min(),
                 MappedType.METADATA_ARRAY_MEAN: mtx.mean()}

    data_group_path = SparseMatrix.ROOT_PATH + 'matrix'
    dt.set_metadata(info_dict, '', True, data_group_path)



def update(input_file):
    """
    In order to avoid segmentation faults when updating a batch of files just
    start every conversion on a different Python process.

    :param input_file: the file that needs to be converted to a newer file storage version.
        This should be a file that still uses TVB 2.0 storage
    """
    if not os.path.isfile(input_file):
        raise FileVersioningException("The input path %s received for upgrading from 2 -> 3 is not a "
                                      "valid file on the disk." % input_file)

    service = ImportService()
    folder, file_name = os.path.split(input_file)
    operation_id = int(os.path.split(folder)[1])
    datatype = service.load_datatype_from_file(folder, file_name, operation_id, move=False)
    if datatype.type == "LocalConnectivity":
        _update_localconnectivity_metadata(datatype)

    root_metadata = datatype.get_metadata()
    root_metadata[TvbProfile.current.version.DATA_VERSION_ATTRIBUTE] = TvbProfile.current.version.DATA_VERSION
    datatype.set_metadata(root_metadata)