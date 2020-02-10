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

import os
import shutil
import tempfile
from tvb.basic.profile import TvbProfile
from werkzeug.utils import secure_filename


def save_temporary_file(file, destination_folder):
    filename = secure_filename(file.filename)
    full_path = os.path.join(destination_folder, filename)
    file.save(full_path)

    return full_path


def get_destination_folder():
    temp_name = tempfile.mkdtemp(dir=TvbProfile.current.TVB_TEMP_FOLDER)
    destination_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, temp_name)

    return destination_folder


def handle_data_file(data_file, destination_folder, view_model_h5, storage_path, number):
    data_file_path = save_temporary_file(data_file, destination_folder)
    file_name = os.path.basename(data_file_path)
    field_name = view_model_h5.view_model.get_upload_files_names()[number]
    upload_field = getattr(view_model_h5, field_name)
    upload_field.store(os.path.join(storage_path, file_name))
    shutil.move(data_file_path, storage_path)

    return data_file_path
