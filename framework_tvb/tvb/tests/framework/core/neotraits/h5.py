import os

import numpy
import pytest
from .data import TestDatatype
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar


class TestDatatypeFileManual(H5File):
    def __init__(self, path):
        super(TestDatatypeFileManual, self).__init__(path)
        self.array_float = DataSet(TestDatatype.array_float, self)
        self.array_int = DataSet(TestDatatype.array_int, self)
        self.scalar_int = Scalar(TestDatatype.scalar_int, self)
        self.scalar_str = Scalar(TestDatatype.scalar_str, self)


class TestDatatypeFile(H5File):
    trait = TestDatatype
    fields = [
        TestDatatype.array_float,
        TestDatatype.array_int,
        TestDatatype.scalar_int,
        TestDatatype.scalar_str
    ]


@pytest.fixture
def tmph5(tmpdir):
    path = os.path.join(str(tmpdir), 'tmp.h5')
    if os.path.exists(path):
        os.remove(path)
    return path


def test_autogenerate_accessors(tmph5):
    f = TestDatatypeFile(tmph5)
    # all declarative attrs generate accessors except the nonmapped one
    assert set(TestDatatype.declarative_attrs) - {'non_mapped_attr'} <= set(f.__dict__)


def test_store_autogen(tmph5, datatypeinstance):
    f = TestDatatypeFile(tmph5)
    f.store(datatypeinstance)


def test_store_manual(tmph5, datatypeinstance):
    f = TestDatatypeFileManual(tmph5)
    f.store(datatypeinstance)


def test_store_load(tmph5, datatypeinstance):
    f = TestDatatypeFile(tmph5)
    datatypeinstance.non_mapped_attr = numpy.array([5.3])
    f.store(datatypeinstance)

    ret = TestDatatype()
    f._load_into(ret)
    # all mapped attributes have been loaded
    assert ret.scalar_int == 42
    assert ret.scalar_str == 'ana'
    assert ret.array_int.shape == (8, 8)
    assert ret.array_float.shape == (100,)
    # this one is not mapped
    assert ret.non_mapped_attr is None

