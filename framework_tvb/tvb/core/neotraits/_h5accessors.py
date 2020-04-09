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

import abc
import json
import uuid
import numpy
import scipy.sparse
import typing
from tvb.basic.neotraits.api import HasTraits, Attr, NArray
from tvb.core.entities.file.exceptions import MissingDataSetException
from tvb.datatypes import equations


class Accessor(object, metaclass=abc.ABCMeta):
    def __init__(self, trait_attribute, h5file, name=None):
        # type: (Attr, H5File, str) -> None
        """
        :param trait_attribute: A traited attribute
        :param h5file: The parent H5file that contains this Accessor
        :param name: The name of the dataset or attribute in the h5 file.
                     It defaults to the name of the associated traited attribute.
                     If the traited attribute is not a member of a HasTraits then
                     it has no name and you have to provide this parameter
        """
        self.owner = h5file
        self.trait_attribute = trait_attribute

        if name is None:
            name = trait_attribute.field_name

        self.field_name = name

        if self.field_name is None:
            raise ValueError('Independent Accessor {} needs a name'.format(self))

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def store(self, val):
        pass

    def __repr__(self):
        cls = type(self)
        return '<{}.{}({}, name="{}")>'.format(
            cls.__module__, cls.__name__, self.trait_attribute, self.field_name
        )


class Scalar(Accessor):
    """
    A scalar in a h5 file that corresponds to a traited attribute.
    Serialized as a global h5 attribute
    """

    def store(self, val):
        # type: (typing.Union[str, int, float]) -> None
        # noinspection PyProtectedMember
        val = self.trait_attribute._validate_set(None, val)
        if val is not None:
            self.owner.storage_manager.set_metadata({self.field_name: val})

    def load(self):
        # type: () -> typing.Union[str, int, float]
        # assuming here that the h5 will return the type we stored.
        # if paranoid do self.trait_attribute.field_type(value)
        metadata = self.owner.storage_manager.get_metadata()
        if self.field_name in metadata:
            return metadata[self.field_name]
        else:
            raise MissingDataSetException(self.field_name)


class Uuid(Scalar):
    def store(self, val):
        # type: (uuid.UUID) -> None
        if val is None and not self.trait_attribute.required:
            # this is an optional reference and it is missing
            return
        # noinspection PyProtectedMember
        if not isinstance(val, uuid.UUID):
            raise TypeError("expected uuid.UUID got {}".format(type(val)))
        # urn is a standard encoding, that is obvious an uuid
        # str(gid) is more ambiguous
        self.owner.storage_manager.set_metadata({self.field_name: val.urn})

    def load(self):
        # type: () -> uuid.UUID
        # TODO: handle inexistent field?
        metadata = self.owner.storage_manager.get_metadata()
        if self.field_name in metadata:
            return uuid.UUID(metadata[self.field_name])
        return None


class DataSetMetaData(object):
    """
    simple container for dataset metadata
    Useful as a cache of global min max values.
    Viewers rely on these for colorbars.
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, min, max, mean):
        self.min, self.max, self.mean = min, max, mean

    @classmethod
    def from_array(cls, array):
        try:
            return cls(min=array.min(), max=array.max(), mean=array.mean())
        except (TypeError, ValueError):
            # likely a string array
            return cls(min=None, max=None, mean=None)

    @classmethod
    def from_dict(cls, dikt):
        return cls(min=dikt['Minimum'], max=dikt['Maximum'], mean=dikt['Mean'])

    def to_dict(self):
        return {'Minimum': self.min, 'Maximum': self.max, 'Mean': self.mean}

    def merge(self, other):
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        self.mean = (self.mean + other.mean) / 2


class DataSet(Accessor):
    """
    A dataset in a h5 file that corresponds to a traited NArray.
    """
    def __init__(self, trait_attribute, h5file, name=None, expand_dimension=-1):
        # type: (NArray, H5File, str, int) -> None
        """
        :param trait_attribute: A traited attribute
        :param h5file: The parent H5file that contains this Accessor
        :param name: The name of the dataset in the h5 file.
                     It defaults to the name of the associated traited attribute.
                     If the traited attribute is not a member of a HasTraits then
                     it has no name and you have to provide this parameter
        :param expand_dimension: An int designating a dimension of the array that may grow.
        """
        super(DataSet, self).__init__(trait_attribute, h5file, name)
        self.expand_dimension = expand_dimension

    def append(self, data, close_file=True, grow_dimension=None):
        # type: (numpy.ndarray, bool, int) -> None
        if not grow_dimension:
            grow_dimension = self.expand_dimension
        self.owner.storage_manager.append_data(
            self.field_name,
            data,
            grow_dimension=grow_dimension,
            close_file=close_file
        )
        # update the cached array min max metadata values
        new_meta = DataSetMetaData.from_array(numpy.array(data))
        meta_dict = self.owner.storage_manager.get_metadata(self.field_name)
        if meta_dict:
            meta = DataSetMetaData.from_dict(meta_dict)
            meta.merge(new_meta)
        else:
            # this must be a new file, nothing to merge, set the new meta
            meta = new_meta
        self.owner.storage_manager.set_metadata(meta.to_dict(), self.field_name)


    def store(self, data):
        # type: (numpy.ndarray) -> None
        # noinspection PyProtectedMember
        data = self.trait_attribute._validate_set(None, data)
        if data is None:
            return

        self.owner.storage_manager.store_data(self.field_name, data)
        # cache some array information
        self.owner.storage_manager.set_metadata(
            DataSetMetaData.from_array(data).to_dict(),
            self.field_name
        )

    def load(self):
        # type: () -> numpy.ndarray
        if not self.trait_attribute.required:
            return self.owner.storage_manager.get_data(self.field_name, ignore_errors=True)
        return self.owner.storage_manager.get_data(self.field_name)

    def __getitem__(self, data_slice):
        # type: (typing.Tuple[slice, ...]) -> numpy.ndarray
        return self.owner.storage_manager.get_data(self.field_name, data_slice)

    @property
    def shape(self):
        # type: () -> typing.Tuple[int]
        return self.owner.storage_manager.get_data_shape(self.field_name)

    def get_cached_metadata(self):
        """
        Returns cached properties of this dataset, like min max mean etc.
        This cache is useful for large, expanding datasets,
        when we want to avoid loading the whole dataset just to compute a max.
        """
        meta = self.owner.storage_manager.get_metadata(self.field_name)
        return DataSetMetaData.from_dict(meta)


class EquationScalar(Accessor):
    """
    An attribute in a h5 file that corresponds to a traited Equation.
    """
    KEY_TYPE = 'type'
    KEY_PARAMETERS = 'parameters'

    def __init__(self, trait_attribute, h5file, name=None):
        # type: (Attr, H5File, str) -> None
        """
        :param trait_attribute: A traited Equation attribute
        :param h5file: The parent H5file that contains this Accessor
        :param name: The name of the dataset in the h5 file.
                     It defaults to the name of the associated traited attribute.
                     If the traited attribute is not a member of a HasTraits then
                     it has no name and you have to provide this parameter
        """
        super(EquationScalar, self).__init__(trait_attribute, h5file, name)

    def store(self, data):
        # type: (Equation) -> None
        data = self.trait_attribute._validate_set(None, data)

        eq_meta_dict = {self.KEY_TYPE: str(type(data).__name__),
                        self.KEY_PARAMETERS: data.parameters}

        self.owner.storage_manager.set_metadata({self.field_name: json.dumps(eq_meta_dict)})

    def load(self):
        # type: () -> Equation
        eq_meta_dict = json.loads(self.owner.storage_manager.get_metadata()[self.field_name])

        if eq_meta_dict is None:
            return eq_meta_dict

        eq_type = eq_meta_dict[self.KEY_TYPE]
        eq_class = getattr(equations, eq_type)
        eq_instance = eq_class()
        parameters_dict = eq_meta_dict[self.KEY_PARAMETERS]
        eq_instance.parameters = parameters_dict
        return eq_instance


class Reference(Uuid):
    """
    A reference to another h5 file
    Corresponds to a contained datatype
    """
    def store(self, val):
        # type: (HasTraits) -> None
        """
        The reference is stored as a gid in the metadata.
        :param val: a datatype or a uuid.UUID gid
        """
        if val is None and not self.trait_attribute.required:
            # this is an optional reference and it is missing
            return
        if isinstance(val, HasTraits):
            val = val.gid
        if not isinstance(val, uuid.UUID):
            raise TypeError("expected uuid.UUId or HasTraits, got {}".format(type(val)))
        super(Reference, self).store(val)


class SparseMatrixMetaData(DataSetMetaData):
    """
    Essential metadata for interpreting a sparse matrix stored in h5
    """

    def __init__(self, minimum, maximum, mean, format, dtype, shape):
        super(SparseMatrixMetaData, self).__init__(minimum, maximum, mean)
        self.dtype = dtype
        self.format = format
        self.shape = shape

    @staticmethod
    def parse_shape(shapestr):
        if not shapestr or shapestr[0] != '(' or shapestr[-1] != ')':
            raise ValueError('can not parse shape "{}"'.format(shapestr))
        frags = shapestr[1:-1].split(',')
        return tuple(int(e) for e in frags)

    @classmethod
    def from_array(cls, mtx):
        return cls(
            minimum=mtx.data.min(),
            maximum=mtx.data.max(),
            mean=mtx.data.mean(),
            format=mtx.format,
            dtype=mtx.dtype,
            shape=mtx.shape,
        )

    @classmethod
    def from_dict(cls, dikt):
        return cls(
            minimum=dikt['Minimum'],
            maximum=dikt['Maximum'],
            mean=dikt['Mean'],
            format=dikt['format'],
            dtype=numpy.dtype(dikt['dtype']),
            shape=cls.parse_shape(dikt['Shape']),
        )

    def to_dict(self):
        return {
            'Minimum': self.min,
            'Maximum': self.max,
            'Mean': self.mean,
            'format': self.format,
            'dtype': self.dtype.str,
            'Shape': str(self.shape),
        }



class SparseMatrix(Accessor):
    """
    Stores and loads a scipy.sparse csc or csr matrix in h5.
    """
    constructors = {'csr': scipy.sparse.csr_matrix, 'csc': scipy.sparse.csc_matrix}

    def store(self, mtx):
        # type: (scipy.sparse.spmatrix) -> None
        # noinspection PyProtectedMember
        mtx = self.trait_attribute._validate_set(None, mtx)
        if mtx is None:
            return
        if mtx.format not in self.constructors:
            raise ValueError('sparse format {} not supported'.format(mtx.format))

        if not isinstance(mtx, scipy.sparse.spmatrix):
            raise TypeError("expected scipy.sparse.spmatrix, got {}".format(type(mtx)))

        self.owner.storage_manager.store_data(
            'data',
            mtx.data,
            where=self.field_name
        )
        self.owner.storage_manager.store_data(
            'indptr',
            mtx.indptr,
            where=self.field_name
        )
        self.owner.storage_manager.store_data(
            'indices',
            mtx.indices,
            where=self.field_name
        )
        self.owner.storage_manager.set_metadata(
            SparseMatrixMetaData.from_array(mtx).to_dict(),
            where=self.field_name
        )

    def get_metadata(self):
        meta = self.owner.storage_manager.get_metadata(self.field_name)
        return SparseMatrixMetaData.from_dict(meta)

    def load(self):
        meta = self.get_metadata()
        if meta.format not in self.constructors:
            raise ValueError('sparse format {} not supported'.format(meta.format))
        constructor = self.constructors[meta.format]
        data = self.owner.storage_manager.get_data(
            'data',
            where=self.field_name,
        )
        indptr = self.owner.storage_manager.get_data(
            'indptr',
            where=self.field_name,
        )
        indices = self.owner.storage_manager.get_data(
            'indices',
            where=self.field_name,
        )
        mtx = constructor((data, indices, indptr), shape=meta.shape, dtype=meta.dtype)
        mtx.sort_indices()
        return mtx



class Json(Scalar):
    """
    A python json like data structure accessor
    This works with simple Attr(list) Attr(dict) List(of=...)
    """

    def __init__(self, trait_attribute, h5file, name=None, json_encoder=None, json_decoder=None):
        super(Json, self).__init__(trait_attribute, h5file, name)
        self.json_encoder = json_encoder
        self.json_decoder = json_decoder

    def store(self, val):
        """
        stores a json in the h5 metadata
        """
        val = json.dumps(val, cls=self.json_encoder)
        self.owner.storage_manager.set_metadata({self.field_name: val})

    def load(self):
        val = self.owner.storage_manager.get_metadata()[self.field_name]
        if self.json_decoder:
            return self.json_decoder().decode(val)
        return json.loads(val)


class JsonFinal(Json):
    """
    A python json like data structure accessor meant to be used with Final(dict)
    """