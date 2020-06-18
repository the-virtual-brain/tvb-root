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

import typing
from functools import partial

from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.h5.projections_h5 import ProjectionMatrixH5
from tvb.adapters.datatypes.h5.sensors_h5 import SensorsH5
from tvb.adapters.datatypes.h5.surface_h5 import SurfaceH5
from tvb.basic.neotraits.api import HasTraits
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neotraits.h5 import H5File
from tvb.core.neotraits.db import HasTraitsIndex
from tvb.datatypes.projections import make_proj_matrix, ProjectionMatrix
from tvb.datatypes.sensors import make_sensors
from tvb.datatypes.surfaces import make_surface


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

    def _determine_proper_datatype_from_h5(self, h5file, subtype):
        dct = {SurfaceH5: partial(make_surface, subtype),
               SensorsH5: partial(make_sensors, subtype),
               ProjectionMatrixH5: partial(make_proj_matrix, subtype)}
        h5file_class = type(h5file)
        return type(dct[h5file_class]())

    def get_datatype_for_h5file(self, h5file):
        # type: (H5File) -> typing.Type[HasTraits]
        subtype = h5file.read_subtype_attr()
        if subtype:
            return self._determine_proper_datatype_from_h5(h5file, subtype)
        return self._datatype_for_h5file[type(h5file)]

    def get_index_for_datatype(self, datatype_class):
        # type: (typing.Type[HasTraits]) -> typing.Type[DataType]
        if datatype_class in self._index_for_datatype:
            return self._index_for_datatype[datatype_class]

        for base in datatype_class.__bases__:
            if base in self._index_for_datatype:
                return self._index_for_datatype[base]
        return DataType

    def _determine_proper_datatype_from_index(self, index, subtype):
        dct = {SurfaceIndex: partial(make_surface, subtype),
               SensorsIndex: partial(make_sensors, subtype),
               ProjectionMatrixIndex: partial(make_proj_matrix, subtype)}
        return type(dct[type(index)]())

    def get_datatype_for_index(self, index):
        # type: (HasTraitsIndex) -> typing.Type[HasTraits]
        subtype = index.get_subtype_attr()
        if subtype:
            return self._determine_proper_datatype_from_index(index, subtype)
        return self._datatype_for_index[type(index)]

    def get_h5file_for_index(self, index_class):
        # type: (typing.Type[DataType]) -> typing.Type[H5File]
        return self._h5file_for_index[index_class]

    def register_datatype(self, datatype_class, h5file_class, datatype_index):
        # type: (HasTraits, H5File, DataType) -> None
        self._h5file_for_datatype[datatype_class] = h5file_class
        self._h5file_for_index[datatype_index] = h5file_class
        self._index_for_datatype[datatype_class] = datatype_index
        self._datatype_for_h5file[h5file_class] = datatype_class
        self._datatype_for_index[datatype_index] = datatype_class
