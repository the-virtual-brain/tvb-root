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

"""
Images have been moved inside the project folder.
This script only updated the folder/files structure of a project.
When executed on a project already stored, an update in DB references might also be needed.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
import shutil

from tvb.storage.storage_interface import StorageInterface


def _rewrite_img_meta(pth, op_id):
    figure_dict = StorageInterface().read_metadata_from_xml(pth)
    figure_dict['file_path'] = op_id + '-' + figure_dict['file_path']
    StorageInterface().write_metadata_in_xml(figure_dict, pth)


def _rename_images(op_id, img_path):
    """
    Place operationId in the stored image names, to make them unique.
    """
    for f in os.listdir(img_path):
        new_name = op_id + '-' + f
        src_pth = os.path.join(img_path, f)
        dst_pth = os.path.join(img_path, new_name)

        if StorageInterface().ends_with_tvb_file_extension(f):
            _rewrite_img_meta(src_pth, op_id)
        os.rename(src_pth, dst_pth)


def _move_folder_content(src_folder, dest_folder):
    for f in os.listdir(src_folder):
        shutil.move(os.path.join(src_folder, f), os.path.join(dest_folder, f))


def update(project_path):
    """
    Images have been moved in the project folder.
    An associated db migration will update file paths in the db.
    """
    new_img_folder = os.path.join(project_path, StorageInterface.IMAGES_FOLDER)
    StorageInterface().check_created(new_img_folder)

    for root, dirs, files in os.walk(project_path):
        in_operation_dir_with_images = StorageInterface.IMAGES_FOLDER in dirs and "Operation.xml" in files

        if in_operation_dir_with_images:
            op_id = os.path.basename(root)
            images_folder = os.path.join(root, StorageInterface.IMAGES_FOLDER)
            _rename_images(op_id, images_folder)
            _move_folder_content(images_folder, new_img_folder)
            os.rmdir(images_folder)
