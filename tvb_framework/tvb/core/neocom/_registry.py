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

import typing
from enum import Enum

from tvb.basic.neotraits.api import HasTraits, TVBEnum
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neotraits.db import HasTraitsIndex
from tvb.core.neotraits.h5 import H5File


class Registry(object):
    """
    A configuration class that holds the one to one relationship
    between datatypes and H5Files that can read/write them to disk
    """

    def __init__(self):
        self._datatype_for_h5file = {}
        self._h5file_for_datatype = {}
        self._h5file_for_index = {}
        self._index_for_datatype = {}
        self._datatype_for_index = {}
        self._index_for_h5file = {}
        self._index_to_subtype_factory = {}

    def get_h5file_for_datatype(self, datatype_class):
        # type: (typing.Type[HasTraits]) -> typing.Type[H5File]
        if datatype_class in self._h5file_for_datatype:
            return self._h5file_for_datatype[datatype_class]
        for base in datatype_class.__bases__:
            if base in self._h5file_for_datatype:
                return self._h5file_for_datatype[base]
        return H5File

    def get_base_datatype_for_h5file(self, h5file_class):
        # type: (typing.Type[H5File]) -> typing.Type[HasTraits]
        return self._datatype_for_h5file[h5file_class]

    def get_datatype_for_h5file(self, h5file):
        # type: (H5File) -> typing.Type[HasTraits]
        base_dt = self._datatype_for_h5file[type(h5file)]
        subtype = h5file.read_subtype_attr()
        if subtype:
            index = self.get_index_for_datatype(base_dt)
            function, enum_class = self._index_to_subtype_factory[index]
            subtype_as_enum = TVBEnum.string_to_enum(list(enum_class), subtype)
            return type(function(subtype_as_enum.value))
        return base_dt

    def get_index_for_datatype(self, datatype_class):
        # type: (typing.Type[HasTraits]) -> typing.Type[DataType]
        if datatype_class in self._index_for_datatype:
            return self._index_for_datatype[datatype_class]

        for base in datatype_class.__bases__:
            if base in self._index_for_datatype:
                return self._index_for_datatype[base]
        return DataType

    def get_datatype_for_index(self, index):
        # type: (HasTraitsIndex) -> typing.Type[HasTraits]
        subtype = index.get_subtype_attr()
        if subtype:
            function, enum_class = self._index_to_subtype_factory[type(index)]
            subtype_as_enum = TVBEnum.string_to_enum(list(enum_class), subtype)
            return type(function(subtype_as_enum.value))
        return self._datatype_for_index[type(index)]

    def get_h5file_for_index(self, index_class):
        # type: (typing.Type[DataType]) -> typing.Type[H5File]
        return self._h5file_for_index[index_class]

    def get_index_for_h5file(self, h5file_class):
        # type: (typing.Type[H5File]) -> typing.Type[DataType]
        return self._index_for_h5file[h5file_class]

    def register_datatype(self, datatype_class, h5file_class, datatype_index, subtype_factory=None, subtype_enum=None):
        # type: (HasTraits, H5File, DataType, callable, Enum) -> None
        self._h5file_for_datatype[datatype_class] = h5file_class
        self._h5file_for_index[datatype_index] = h5file_class
        self._index_for_datatype[datatype_class] = datatype_index
        self._datatype_for_h5file[h5file_class] = datatype_class
        self._datatype_for_index[datatype_index] = datatype_class
        self._index_for_h5file[h5file_class] = datatype_index
        self._index_to_subtype_factory[datatype_index] = (subtype_factory, subtype_enum)
