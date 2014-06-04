# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
This module contains basic reading mechanism for default DataType fields.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

try:
    H5PY_SUPPORT = True
    import h5py as hdf5
except Exception:
    H5PY_SUPPORT = False

import os
import numpy
import zipfile
from scipy import io as scipy_io
from tvb.basic.logger.builder import get_logger



class H5Reader():
    """
    Read one or many numpy arrays from a H5 file.
    """

    def __init__(self, h5_path):

        self.logger = get_logger(__name__)
        if H5PY_SUPPORT:
            self.hfd5_source = hdf5.File(h5_path, 'r', libver='latest')
        else:
            self.logger.warning("You need h5py properly installed in order to load from a HDF5 source.")


    def read_field(self, field):

        try:
            return self.hfd5_source['/' + field][()]
        except Exception:
            self.logger.exception("Could not read from %s field" % field)
            return numpy.array([])



class FileReader():
    """
    Read one or multiple numpy arrays from a text/bz2 file.
    """

    def __init__(self, file_path):

        self.logger = get_logger(__name__)
        self.file_path = file_path
        self.file_stream = file_path


    def read_array(self, dtype=numpy.float64, skip_rows=0, use_cols=None, matlab_data_name=None):

        self.logger.debug("Starting to read from: " + str(self.file_path))

        try:
            # Try to read H5:
            if self.file_path.endswith('.h5'):
                self.logger.error("Not yet implemented read from a ZIP of H5 files!")
                return numpy.array([])

            # Try to read NumPy:
            if self.file_path.endswith('.txt') or self.file_path.endswith('.txt.bz2'):
                return self._read_text(self.file_stream, dtype, skip_rows, use_cols)

            if self.file_path.endswith('.npz'):
                return numpy.load(self.file_stream)

            # Try to read Matlab format:
            return self._read_matlab(self.file_stream, matlab_data_name)

        except Exception:
            self.logger.exception("Could not read from %s file" % self.file_path)
            return numpy.array([])


    def _read_text(self, file_stream, dtype, skip_rows, use_cols):

        array_result = numpy.loadtxt(file_stream, dtype=dtype, skiprows=skip_rows, usecols=use_cols)
        return array_result


    def _read_matlab(self, file_stream, matlab_data_name=None):

        if self.file_path.endswith(".mtx"):
            return scipy_io.mmread(file_stream)

        if self.file_path.endswith(".mat"):
            matlab_data = scipy_io.matlab.loadmat(file_stream)
            return matlab_data[matlab_data_name]



class ZipReader():
    """
    Read one or many numpy arrays from a ZIP archive.
    """

    def __init__(self, zip_path):

        self.logger = get_logger(__name__)
        self.zip_archive = zipfile.ZipFile(zip_path)


    def read_array_from_file(self, file_name, dtype=numpy.float64, skip_rows=0, use_cols=None, matlab_data_name=None):

        file_reader = FileReader(file_name)

        matching_file_name = None
        for actual_name in self.zip_archive.namelist():
            if actual_name.endswith(file_name):
                matching_file_name = actual_name

        if matching_file_name is not None:
            file_reader.file_stream = self.zip_archive.open(matching_file_name, 'rU')
            return file_reader.read_array(dtype, skip_rows, use_cols, matlab_data_name)
        else:
            self.logger.warning("File %s not found in ZIP." % file_name)

        return numpy.array([])



def try_get_absolute_path(relative_module, file_suffix):
    """
    :param relative_module: python module to be imported. When import of this fails, we will return the file_suffix
    :param file_suffix: In case this is already an absolute path, return it immediately,
        otherwise append it after the module path
    :return: Try to build an absolute path based on a python module and a file-suffix
    """

    result_full_path = file_suffix

    if not os.path.isabs(file_suffix):

        try:
            module_import = __import__(relative_module, globals(), locals(), ["__init__"])
            result_full_path = os.path.join(os.path.dirname(module_import.__file__), file_suffix)

        except ImportError:
            LOG = get_logger(__name__)
            LOG.exception("Could not import tvb_data Python module for default data-set!")

    return result_full_path
