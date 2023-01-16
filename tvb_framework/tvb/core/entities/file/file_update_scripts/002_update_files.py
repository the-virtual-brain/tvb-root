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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
from datetime import datetime
import os
import sys
import importlib
import tables
import h5py
import numpy
from tvb.basic.profile import TvbProfile
from tvb.core.utils import string2bool, string2date, date2string
from tvb.storage.h5.file.exceptions import FileVersioningException

PYTHON_EXE_PATH = TvbProfile.current.PYTHON_INTERPRETER_PATH
DATA_BUFFER_SIZE = 50000000 / 8  # 500 MB maximum read at once (just assume worst case float64)

# ---------------------- TVB 1.0 Specific constants and functions start here --------------------------
# We duplicate these constants here since they were the ones used in TVB 1.0 and 
# this upgrade script needs to be able to convert any data that was created in 1.0 to 1.0.1.
# Since this script needs to execute properly even in a latter version (e.g. 2.0) it's not
# feasible to use any constants currently declared in TVB, since those might change over time.
BOOL_VALUE_PREFIX = "bool:"
DATETIME_VALUE_PREFIX = "datetime:"
DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
COMPLEX_TIME_FORMAT = '%Y-%m-%d,%H-%M-%S.%f'
SIMPLE_TIME_FORMAT = "%m-%d-%Y"
TVB_ATTRIBUTE_PREFIX = "TVB_"
DATA_VERSION_ATTRIBUTE = "Data_version"


def _serialize_value(value):
    """
    This method takes a value which will be stored as meta-data and 
    apply some transformation if necessary
    
    :param value:  value which is planned to be stored
    :returns: value to be stored
    
    NOTE: this method was a part of TVB 1.0 hdf5storage manager, but since this
    script needs to be independent of current storage manager, we duplicate it here. 
    """
    if value is None:
        return ''
    # Transform boolean to string and prefix it
    if isinstance(value, bool):
        return BOOL_VALUE_PREFIX + str(value)
    # Transform date to string and append prefix
    elif isinstance(value, datetime):
        return DATETIME_VALUE_PREFIX + date2string(value, date_format=DATE_TIME_FORMAT)
    else:
        return value


def _deserialize_value(value):
    """
    This method takes value loaded from H5 file and transform it to TVB data. 
    
    :param value: the value that was read from the H5 file
    :returns: a TVB specific deserialized value of the input
    
    NOTE: this method was a part of TVB 1.0 hdf5storage manager, but since this
    script needs to be independent of current storage manager, we duplicate it here. 
    """
    if value is not None:
        if isinstance(value, numpy.string_):
            if len(value) == 0:
                value = None
            else:
                value = str(value)
        if isinstance(value, str):
            if value.startswith(BOOL_VALUE_PREFIX):
                # Remove bool prefix and transform to bool
                return string2bool(value[len(BOOL_VALUE_PREFIX):])
            if value.startswith(DATETIME_VALUE_PREFIX):
                # Remove datetime prefix and transform to datetime
                return string2date(value[len(DATETIME_VALUE_PREFIX):], date_format=DATE_TIME_FORMAT)
    return value
# ---------------------- TVB 1.0 Specific constants and functions end here --------------------------


def __upgrade_file(input_file_name, output_file_name):
    """
    This method does any required processing in order to convert an input file stored in
    TVB 1.0 format into an output_file of TVB 2.0 format. 
    
    NOTE: This should not be used directly since the simultaneous use of pyTables and h5py 
    causes segmentation faults on some setups (Debian 32/65, Fedora 64, Windows 64) on file
    open/close. (Probably caused by some GIL / C level incompatibilities). Instead of this 
    use the `upgrade(file_name)` which will call this method in a separate Python process.
    
    :param input_file_name: the path to a input *.h5 file from TVB 1.0 using pyTables format
        for storage
    :param output_file_name: the path to a output *.h5 that will be written in h5py TVB 1.0.1
        specific format
    """
    tables_h5_file = tables.openFile(input_file_name, 'r')
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
    h5py_h5_file = h5py.File(output_file_name, 'a')
    # Iterate through all pyTables nodes
    for tables_node in tables_h5_file.walkNodes():
        node_path = tables_node._v_pathname.replace('/', '')
        node_metadata = {}
        # Get meta-data from the pyTables node. This does not change for root/group/Carray nodes
        all_meta_keys = tables_node._v_attrs._f_list('user')
        for meta_key in all_meta_keys:
            new_key = meta_key
            value = tables_h5_file.getNodeAttr(tables_node, meta_key)
            node_metadata[new_key] = _deserialize_value(value)
        if tables_node.__class__ is tables.group.RootGroup:
            # For the root the node is already created in the h5py equivalent
            h5py_node = h5py_h5_file['/'] 
        elif tables_node.__class__ is tables.group.Group:
            # For groups just create an empty datas-et since it's easier to handle
            # than sub-groups.
            h5py_node = h5py_h5_file.create_dataset(node_path, (1,))
        else:
            # We have a standard node (Carray), compute based on the shape if it will
            # fit in the DATA_BUFFER_SIZE we set or we need to read/write by chunks.
            node_shape = tables_node.shape
            max_dimension = 0
            total_size = 1
            for idx, val in enumerate(node_shape):
                if val > node_shape[max_dimension]:
                    max_dimension = idx
                total_size = total_size * val
            if total_size <= DATA_BUFFER_SIZE:
                # We did not pass our buffer size, so it's save to just read/write the whole data at once
                node_data = tables_node.read()
                h5py_node = h5py_h5_file.create_dataset(node_path, data=node_data, 
                                                        shape=node_data.shape, dtype=node_data.dtype)
            else:
                # We need to read in chunks. Set the dimension that is growable to None
                node_shape_list = list(node_shape)
                node_shape_list[max_dimension] = None
                h5py_node = h5py_h5_file.create_dataset(node_path, shape=node_shape, maxshape=tuple(node_shape_list))
                slice_size = max(int(DATA_BUFFER_SIZE * node_shape[max_dimension] / total_size), 1)
                full_slice = slice(None, None, None)
                data_slice = [full_slice for _ in node_shape]
                for idx in range(0, node_shape[max_dimension], slice_size):
                    specific_slice = slice(idx, idx + slice_size, 1)
                    data_slice[max_dimension] = specific_slice
                    tables_data = tables_node[tuple(data_slice)]
                    h5py_node = h5py_h5_file[node_path]
                    h5py_node[tuple(data_slice)] = tables_data
        for meta_key in node_metadata:
            processed_value = _serialize_value(node_metadata[meta_key])
            h5py_node.attrs[meta_key] = processed_value
    h5py_h5_file['/'].attrs[TVB_ATTRIBUTE_PREFIX + DATA_VERSION_ATTRIBUTE] = 2
    tables_h5_file.close()
    # Reloading h5py seems to fix the segmentation fault that used to appear.
    importlib.reload(h5py)
    h5py_h5_file.close()
    

def update(input_file, burst_match_dict=None):
    """
    In order to avoid segmentation faults when updating a batch of files just
    start every conversion on a different Python process.
    
    :param input_file: the file that needs to be converted to a newer file storage version.
        This should be a file that still uses TVB 1.0 storage (pyTables)
    """
    # Just to avoid any problems about renaming open files, do a rename from the start 
    # and if case of a fault in the os.system call just rename back and remove the output file.
    if not os.path.isfile(input_file):
        raise FileVersioningException("The input path %s received for upgrading from 1 -> 2 is not a "
                                      "valid file on the disk." % input_file)
    # Use a file-path with no spaces both for the temporary file and the input file
    # that is passed to the os.system call and just rename to original file at the 
    # end of the processing to avoid any problems with parameters passed to os.system.
    input_file_no_spaces = input_file.replace(' ', '')
    path_to, file_name = os.path.split(input_file_no_spaces)
    tmp_convert_file = os.path.join(path_to, 'tmp_' + file_name)
    os.rename(input_file, tmp_convert_file)
    ok_status = os.system(PYTHON_EXE_PATH + ' -m %s %s %s' % (__name__, tmp_convert_file, input_file_no_spaces))
    if ok_status == 0:
        # Call finished successfully
        os.remove(tmp_convert_file)
        os.rename(input_file_no_spaces, input_file)
    else:
        # Call failed for some reason, just rename back the input file
        os.rename(tmp_convert_file, input_file)
        raise FileVersioningException("Something went wrong during the upgrade to file %s." % input_file)
    

### This main is important, and used by the update() method from above.
### Do not drop this  __main__ 
if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise FileVersioningException("Usage is `python -m tvb.core.entities.file.file_update_scripts.002_update_files"
                                      " input_file_name output_file_name`.")
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    __upgrade_file(input_file, output_file)
