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

import shutil
import os
import sys

from tvb.basic.profile import TvbProfile
from tvb.config.init.initializer import initialize
from tvb.core.entities.file.files_update_manager import FilesUpdateManager


if __name__ == '__main__':
    """
    Script written for testing the migration from version 1.5.8 to 2.1.0.
    """

    # set web profile
    TvbProfile.set_profile(TvbProfile.WEB_PROFILE)

    # migrate the database and h5 files
    h5_migrating_thread = initialize()

    # wait for thread to finish before processing
    h5_migrating_thread.join()

    # copy files in tvb_root folder so Jenkins can find them
    EXTERNALS_FOLDER_PARENT = os.path.dirname(TvbProfile.current.BIN_FOLDER)
    shutil.copytree(TvbProfile.current.TVB_LOG_FOLDER, os.path.join(EXTERNALS_FOLDER_PARENT, 'logs'))

    # test if there are any files which were not migrated
    number_of_unmigrated_files = len(FilesUpdateManager.get_all_h5_paths())
    if number_of_unmigrated_files != 0:
        sys.exit(-1)
    sys.exit(0)
