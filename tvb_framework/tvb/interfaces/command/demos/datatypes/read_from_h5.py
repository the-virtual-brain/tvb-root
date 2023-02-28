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
Demo script on how to read a TVB H5 file

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import sys

from tvb.core.entities.model.model_operation import Operation
from tvb.core.entities.storage import dao
from tvb.core.services.import_service import ImportService


def read_h5(full_paths):
    # We need to load the DataType in the context of an operation, and a project
    all_projects = dao.get_all_projects()
    all_operations = dao.get_generic_entity(Operation, all_projects[0].id, "fk_launched_in")

    service = ImportService()
    results = []

    for full_path in full_paths:
        # The actual read of H5:
        datatype = service.load_datatype_from_file(full_path, all_operations[0].id)

        print("We've build DataType: [%s]" % datatype.__class__.__name__, datatype)
        results.append(datatype)

    return results


if __name__ == '__main__':
    from tvb.interfaces.command.lab import *

    if len(sys.argv) > 1:
        FILE_PATHS = sys.argv[1:]
    else:
        import tvb_data.h5

        dir_name = os.path.dirname(tvb_data.h5.__file__)
        # TODO: Fix tvb_data h5 files
        FILE_PATHS = [os.path.join(dir_name, "Connectivity_74.h5"),
                      os.path.join(dir_name, "TimeSeriesRegion.h5")]

    print("We will try to read from H5: " + str(FILE_PATHS))

    read_h5(FILE_PATHS)
