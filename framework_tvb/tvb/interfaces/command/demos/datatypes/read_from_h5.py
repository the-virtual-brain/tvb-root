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
Demo script on how to read a TVB H5 file

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

import os
import sys
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.services.import_service import ImportService


def read_h5(full_paths):

    # We need to load the DataType in the context of an operation, and a project
    all_projects = dao.get_all_projects()
    all_operations = dao.get_generic_entity(model.Operation, all_projects[0].id, "fk_launched_in")

    service = ImportService()
    results = []

    for full_path in full_paths:
        # The actual read of H5:
        datatype = service.load_datatype_from_file(full_path, all_operations[0].id)

        print("We've build DataType: [%s]" % datatype.__class__.__name__, datatype)
        results.append(datatype)

    return results



if __name__ == '__main__':

    if len(sys.argv) > 1:
        FILE_PATHS = sys.argv[1:]
    else:
        import tvb_data.h5
        dir_name = os.path.dirname(tvb_data.h5.__file__)
        FILE_PATHS = [os.path.join(dir_name, "Connectivity_74.h5"),
                      os.path.join(dir_name, "TimeSeriesRegion.h5")]

    print("We will try to read from H5: " + str(FILE_PATHS))

    read_h5(FILE_PATHS)

