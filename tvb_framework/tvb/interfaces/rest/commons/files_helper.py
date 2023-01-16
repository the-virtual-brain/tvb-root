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

import os
import tempfile

from tvb.basic.profile import TvbProfile
from werkzeug.utils import secure_filename

CHUNK_SIZE = 128


def save_file(file_path, response):
    with open(file_path, 'wb') as local_file:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                local_file.write(chunk)
    return file_path


def create_temp_folder():
    temp_name = tempfile.mkdtemp(dir=TvbProfile.current.TVB_TEMP_FOLDER)
    folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, temp_name)

    return folder


def save_temporary_file(file, destination_folder=None):
    filename = secure_filename(file.filename)
    if destination_folder is None:
        destination_folder = create_temp_folder()
    full_path = os.path.join(destination_folder, filename)
    file.save(full_path)

    return full_path
