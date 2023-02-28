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

import numpy
from tvb.basic.neotraits.api import Attr, NArray
from .data import FooDatatype, BarDatatype, BazDataType, PropsDataType
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Reference


class BazFile(H5File):
    def __init__(self, path):
        super(BazFile, self).__init__(path)
        self.miu = DataSet(BazDataType.miu, self)
        self.scalar_str = Scalar(BazDataType.scalar_str, self)


class FooFile(H5File):
    def __init__(self, path):
        super(FooFile, self).__init__(path)
        self.array_float = DataSet(FooDatatype.array_float, self)
        self.array_int = DataSet(FooDatatype.array_int, self)
        self.scalar_int = Scalar(FooDatatype.scalar_int, self)
        self.abaz = Reference(FooDatatype.abaz, self)


class BarFile(FooFile):
    # inheritance is flattened in the same file
    def __init__(self, path):
        super(BarFile, self).__init__(path)
        self.array_str = DataSet(BarDatatype.array_str, self)


class Independent(H5File):
    def __init__(self, path):
        super(Independent, self).__init__(path)
        self.scalar_int = Scalar(Attr(int), self, name='scalar_int')
        self.array_float = DataSet(NArray(), self, name='floating_leaves')


def test_independent_h5file(tmph5factory):
    with Independent(tmph5factory()) as f:
        f.scalar_int.store(3)
        f.array_float.store(numpy.eye(4, dtype=float))

        numpy.testing.assert_equal(f.array_float.load(), numpy.eye(4))

        # note that this does nothing as the attributes in the
        # Independent datasets have no field_name
        # because they are not bound to a datatype
        f.store(BazDataType())


def test_accessors_created_for_all_declarative_attributes(tmph5factory):
    f = BazFile(tmph5factory())
    assert set(BazDataType.declarative_attrs) <= set(f.__dict__)


def test_simple_store_load(tmph5factory):
    baz = BazDataType(miu=numpy.array([0.0, 1.0, 2.0]), scalar_str='topol')
    path = tmph5factory()
    f = BazFile(path)
    f.store(baz)
    f.close()

    ret = BazDataType()
    assert ret.scalar_str is None
    f = BazFile(path)
    f.load_into(ret)
    assert ret.scalar_str == 'topol'
    assert numpy.all(ret.miu == [0.0, 1.0, 2.0])


def test_dataset_metadata(tmph5factory):
    baz = BazDataType(miu=numpy.array([0.0, 1.0, 2.0]), scalar_str='topol')
    path = tmph5factory()

    with BazFile(path) as f:
        f.store(baz)

    with BazFile(path) as f:
        assert f.miu.get_cached_metadata().max == 2.0



def test_aggregate_store(tmph5factory, fooFactory):
    foo = fooFactory()
    foofile = FooFile(tmph5factory('foo-{}.h5'.format(foo.gid)))
    foofile.store(foo)
    foofile.close()
    bazfile = BazFile(tmph5factory('baz-{}.h5'.format(foo.abaz.gid)))
    bazfile.store(foo.abaz)
    bazfile.close()


def test_store_load_inheritance(tmph5factory, barFactory):
    bar = barFactory()
    path = tmph5factory()
    with BarFile(path) as barfile:
        barfile.store(bar)

    with BarFile(path) as barfile:
        ret = BarDatatype()
        barfile.load_into(ret)
        assert ret.scalar_int == bar.scalar_int


def test_store_load_preserves_gid(tmph5factory, barFactory):
    bar = barFactory()
    path = tmph5factory()
    with BarFile(path) as barfile:
        barfile.store(bar)

    with BarFile(path) as barfile:
        ret = BarDatatype()
        barfile.load_into(ret)
        assert ret.gid == bar.gid


def test_append(tmph5factory):
    pth = tmph5factory()

    with BazFile(pth) as f:
        for i in range(4):
            f.miu.append(i * numpy.eye(2))

        meta = f.miu.get_cached_metadata()
        assert meta.min == 0
        assert meta.max == 3


def test_props_datatype_file(tmph5factory):

    datatype = PropsDataType(n_node=3)
    datatype.weights = numpy.eye(3)
    datatype.validate()


    class PropsDataTypeFile(H5File):
        def __init__(self, path):
            super(PropsDataTypeFile, self).__init__(path)

            self.n_node = Scalar(PropsDataType.n_node, self)
            # You cannot define Accessors for trait_properties
            # You have to manually map them, using datatype independent accessors
            # breaks: self.weights = DataSet(PropsDataType.weights, self)
            self.weights = DataSet(NArray(), self, name='weights')
            # As any independent accessors weights will be ignored by H5File.load_into, H5File.store

            # Properties have no obvious serialization semantics.
            # They might not be writable or may need a specific initalisation order
            # in this case setting n_node before weights

            self.is_directed = Scalar(Attr(bool), self, name='is_directed')
            self.once = DataSet(NArray(), self, name='once')

        def store(self, datatype, scalars_only=False):
            super(PropsDataTypeFile, self).store(datatype, scalars_only=scalars_only)
            # handle the trait_properties manually
            self.weights.store(datatype.weights)
            # is_directed is read only, We choose to store it in the h5 as an example
            # it will never be read from there.
            # One might want to store these in h5 so that manually opened h5 are more informative
            self.is_directed.store(datatype.is_directed)
            # self.once is read only, no need to store

        def load_into(self, datatype):
            super(PropsDataTypeFile, self).load_into(datatype)
            # n_node is loaded, so weights will not complain about missing shape info
            datatype.weights = self.weights.load()
            # datatype.is_directed is read only
            # datatype.once is read only as well


    ret = PropsDataType()
    path = tmph5factory()

    with PropsDataTypeFile(path) as f:
        f.store(datatype)

    with PropsDataTypeFile(path) as f:
        f.load_into(ret)
