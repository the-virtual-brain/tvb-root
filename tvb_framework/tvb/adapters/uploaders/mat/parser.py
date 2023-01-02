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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import numpy
import scipy.io


def read_nested_mat_structure(m, structure_path):
    """
    Reads data from a hierarchical structure array.
    If object arrays of shape (1,1) are found they are automatically flattened.
    :param m: A numpy structure array originating from a matlab mat file
    :param structure_path: A dot delimited path of field names: topfield.child.leaf
    :return: The leaf
    """
    structure_path = structure_path.strip()
    nested_fields = structure_path.split('.')

    if not structure_path:
        return m
    if '' in nested_fields:
        raise ValueError("bad path: '%s' " % structure_path)

    try:
        for field_name in nested_fields:
            # unwrap object arrays containers of shape 1, 1
            m = m[field_name]
            if issubclass(m.dtype.type, numpy.object_) and m.shape == (1, 1):
                m = m[0, 0]
    except ValueError as ex:
        raise ValueError("missing field: %s" % ex[0])

    return m


def read_nested_mat_file(data_file, dataset_name, structure_path):
    """
    Reads data from deep structures from a .mat file
    :param data_file: path to the mat file
    :param dataset_name: matlab variable name
    :param structure_path: A dot delimited path of field names: topfield.child.leaf
    :return: the leaf data
    """
    mat = scipy.io.loadmat(data_file)
    try:
        return read_nested_mat_structure(mat[dataset_name], structure_path)
    except KeyError as ex:
        raise KeyError("could not find: %s" % ex[0])