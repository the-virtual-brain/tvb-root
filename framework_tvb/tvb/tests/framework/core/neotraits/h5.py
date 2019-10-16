import os

import numpy
import pytest
from .data import FooDatatype, BooDatatype
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar


class FooFileManual(H5File):
    def __init__(self, path):
        super(FooFileManual, self).__init__(path)
        self.array_float = DataSet(FooDatatype.array_float, self)
        self.array_int = DataSet(FooDatatype.array_int, self)
        self.scalar_int = Scalar(FooDatatype.scalar_int, self)
        self.scalar_str = Scalar(FooDatatype.scalar_str, self)


class FooFile(H5File):
    trait = FooDatatype
    fields = [
        FooDatatype.array_float,
        FooDatatype.array_int,
        FooDatatype.scalar_int,
        FooDatatype.scalar_str
    ]


class BooFile(FooFile):
    trait = BooDatatype
    fields = [
        BooDatatype.array_str
    ]



@pytest.fixture
def tmph5(tmpdir):
    path = os.path.join(str(tmpdir), 'tmp.h5')
    if os.path.exists(path):
        os.remove(path)
    return path


def test_autogenerate_accessors(tmph5):
    f = FooFile(tmph5)
    # all declarative attrs generate accessors except the nonmapped one
    assert set(FooDatatype.declarative_attrs) - {'non_mapped_attr'} <= set(f.__dict__)


def test_store_autogen(tmph5, fooFactory):
    f = FooFile(tmph5)
    f.store(fooFactory())


def test_store_manual(tmph5, fooFactory):
    f = FooFileManual(tmph5)
    f.store(fooFactory())


def test_store_load(tmph5, fooFactory):
    f = FooFile(tmph5)
    datatypeinstance = fooFactory()
    datatypeinstance.non_mapped_attr = numpy.array([5.3])
    f.store(datatypeinstance)

    ret = FooDatatype()
    f._load_into(ret)
    # all mapped attributes have been loaded
    assert ret.scalar_int == 42
    assert ret.scalar_str == 'ana'
    assert ret.array_int.shape == (8, 8)
    assert ret.array_float.shape == (100,)
    # this one is not mapped
    assert ret.non_mapped_attr is None


def test_load_store_inheritance(tmph5, booFactory):
    boo = booFactory()
    f = BooFile(tmph5)
    f.store(boo)

    ret = BooDatatype()
    f._load_into(ret)
    assert ret.array_str.tolist() == ['ana', 'are', 'mere']
    assert ret.scalar_str == 'ana'


