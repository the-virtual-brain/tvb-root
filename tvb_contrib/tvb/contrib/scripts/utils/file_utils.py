# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
File writing/reading and manipulations

.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

import os
from datetime import datetime
import glob
import shutil


def ensure_unique_file(parent_folder, filename):
    final_path = os.path.join(parent_folder, filename)
    while os.path.exists(final_path):
        filename = input("\n\nFile %s already exists. Enter a different name: " % final_path)
        final_path = os.path.join(parent_folder, filename)
    return final_path


def change_filename_or_overwrite(path, overwrite=True):
    if overwrite:
        if os.path.exists(path):
            os.remove(path)
        return path

    parent_folder = os.path.dirname(path)
    while os.path.exists(path):
        filename = \
            input("\n\nFile %s already exists. Enter a different name or press enter to overwrite file: " % path)
        if filename == "":
            overwrite = True
            break

        path = os.path.join(parent_folder, filename)

    if overwrite:
        os.remove(path)

    return path


def insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))


def wildcardit(name, front=True, back=True):
    out = str(name)
    if front:
        out = "*" + out
    if back:
        out = out + "*"
    return out


def delete_all_files_in_folder(folderpath):
    if os.path.isdir(folderpath):
        for filename in os.listdir(folderpath):
            file_path = os.path.join(folderpath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def delete_folder_safely(folderpath):
    if os.path.isdir(folderpath):
        try:
            shutil.rmtree(folderpath)
        except Exception as e:
            print('Failed to delete %s.\nReason: %s' % (folderpath, e))


def change_filename_or_overwrite_with_wildcard(path, overwrite=True):
    wild_path = path + "*"
    existing_files = glob.glob(path + "*")
    if len(existing_files) > 0:
        if overwrite:
            for file in existing_files:
                if os.path.exists(file):
                    os.remove(file)
            return path
        else:
            print("The following files already exist for base paths " + wild_path + " !: ")
            for file in existing_files:
                print(file)
            filename = input("\n\nEnter a different name or press enter to overwrite files: ")
            if filename == "":
                return change_filename_or_overwrite_with_wildcard(path, overwrite=True)
            else:
                parent_folder = os.path.dirname(path)
                path = os.path.join(parent_folder, filename)
                return change_filename_or_overwrite_with_wildcard(path, overwrite)
    else:
        return path


def move_overwrite_files_to_folder_with_wildcard(folder, path_wildcard):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for file in glob.glob(path_wildcard):
        if os.path.isfile(file):
            filepath = os.path.join(folder, os.path.basename(file))
            shutil.move(file, filepath)


def write_metadata_in_xml(meta_dict, h5_file, key_date, key_version, path="/"):
    root = h5_file[path].attrs
    root[key_date] = str(datetime.now())
    root[key_version] = 2
    for key, val in meta_dict.items():
        root[key] = val


def safe_makedirs(folder):
    # solution based on https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    if not os.path.isdir(folder):
        try:
            # This should work for Python >= 3.2
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:  # Python ≥ 2.5
            try:
                os.makedirs(folder)
            except:
                try:  # just in case errno cannot be imported...
                    import errno
                    if exc.errno == errno.EEXIST and os.path.isdir(folder):
                        pass
                    else:
                        raise
                except:
                    pass