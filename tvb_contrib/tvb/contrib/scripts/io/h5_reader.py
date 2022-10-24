# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

import os

import h5py

from tvb.basic.readers import H5Reader as H5ReaderBase

from .base import Base
from .h5_writer import H5Writer


class H5GroupHandlers(object):

    H5_TYPE_ATTRIBUTE = H5Writer.H5_TYPE_ATTRIBUTE
    H5_SUBTYPE_ATTRIBUTE = H5Writer.H5_SUBTYPE_ATTRIBUTE
    H5_TYPES_ATTRUBUTES = [H5_TYPE_ATTRIBUTE, H5_SUBTYPE_ATTRIBUTE]

    def __init__(self, h5_subtype_attribute):
        if h5_subtype_attribute is not None:
            self.H5_SUBTYPE_ATTRIBUTE = h5_subtype_attribute

    def read_dictionary_from_group(self, group):  # , type=None
        dictionary = dict()
        for dataset in group.keys():
            try:
                value = group[dataset][()]
            except:
                try:
                    value = self.read_dictionary_from_group(group[dataset])
                except:
                    value = None
            dictionary.update({dataset: value})
        for attr in group.attrs.keys():
            if attr not in self.H5_TYPES_ATTRUBUTES:
                dictionary.update({attr: group.attrs[attr]})
        # if type is None:
        #     type = group.attrs[H5Reader.H5_SUBTYPE_ATTRIBUTE]
        # else:
        return dictionary


class H5Reader(H5ReaderBase, Base):

    H5_TYPE_ATTRIBUTE = H5Writer.H5_TYPE_ATTRIBUTE
    H5_SUBTYPE_ATTRIBUTE = H5Writer.H5_SUBTYPE_ATTRIBUTE
    H5_TYPES_ATTRUBUTES = [H5_TYPE_ATTRIBUTE, H5_SUBTYPE_ATTRIBUTE]

    hfd5_source = None

    def __init__(self, h5_path):
        super(H5Reader, self).__init__(h5_path)
        self.h5_path = h5_path

    @property
    def _hdf_file(self):
        return self.hfd5_source

    def _set_hdf_file(self, hfile):
        self.hfd5_source = hfile

    @property
    def _fmode(self):
        return "r"

    @property
    def _mode(self):
        return "read"

    @property
    def _mode_past(self):
        return "read"

    @property
    def _to_from(self):
        return "from"

    def _open_file(self, type_name=""):
        if not os.path.isfile(self.h5_path):
            raise ValueError("%s file %s does not exist" % (type_name, self.h5_path))
        super(H5Reader, self)._open_file(type_name)

    def read_dictionary(self, path=None, close_file=True):  # type=None,
        """
        :return: dict
        """
        dictionary = dict()
        self._assert_file(path, "Dictionary")
        try:
            dictionary = H5GroupHandlers(self.H5_SUBTYPE_ATTRIBUTE).read_dictionary_from_group(h5_file)  # , type
            self._log_success_or_warn(None, "Dictionary")
        except Exception as e:
            self._log_success_or_warn(e, "Dictionary")
        self._close_file(close_file)
        return dictionary

    def read_list_of_dicts(self, path=None, close_file=True):  # type=None,
        self._assert_file(path, "List of Dictionaries")
        list_of_dicts = []
        id = 0
        h5_group_handlers = H5GroupHandlers(self.H5_SUBTYPE_ATTRIBUTE)
        try:
            while 1:
                try:
                    dict_group = h5_file[str(id)]
                except:
                    break
                list_of_dicts.append(h5_group_handlers.read_dictionary_from_group(dict_group))  # , type
                id += 1
            self._log_success_or_warn(None,  "List of Dictionaries")
        except Exception as e:
            self._log_success_or_warn(e, "List of Dictionaries")
        self._close_file(close_file)
        return list_of_dicts
