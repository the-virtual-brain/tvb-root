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
Demo script for console profile which is showing how a project import can be done from the command line.
Later on, this project will be available from the web-interface.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sys import argv

from tvb.core.entities import model
from tvb.core.services.import_service import ImportService
from tvb.core.services.user_service import UserService


def run_import(project_path):
    ## If we would know a UserID to have as admin, next step would not be necessary.
    ## Make sure at least one user exists in TVB DB:
    user_service = UserService()
    admins = user_service.get_administrators()

    if admins:
        admin = admins[0]
    else:
        ## No Admin user was found, we will create one
        user_service.create_user("admin", 'display name', "pass", role=model.ROLE_ADMINISTRATOR,
                                 email="info@thevirtualbrain.org", validated=True, skip_import=True)
        admin = user_service.get_administrators()[0]

    ## Do the actual import of a project from ZIP:
    import_service = ImportService()
    import_service.import_project_structure(project_path, admin.id)

    print("Project imported successfully. Check the Web UI!")


if __name__ == '__main__':
    from tvb.config.init.initializer import command_initializer

    command_initializer(skip_import=True)

    if len(argv) < 2:
        print("No Project path given!!!")
        exit(1)

    PROJECT_PATH = str(argv[1])

    print("We will try to import project at path " + PROJECT_PATH)

    run_import(PROJECT_PATH)
