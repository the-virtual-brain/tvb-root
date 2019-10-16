import os

import numpy
import pytest
from .data import FooDatatype, BarDatatype, BazDataType
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Reference


@pytest.fixture
def tmph5factory(tmpdir):
    def build(pth='tmp.h5'):
        path = os.path.join(str(tmpdir), pth)
        if os.path.exists(path):
            os.remove(path)
        return path
    return build



class BazFile(H5File):
    # trait is declared. This is an automated mapping
    trait = BazDataType
    # automatic mapping creates metadata entries for scalar trait attributes
    # datasets for traited ndarrays
    # and reference fields for aggregated HasTraits


class FooFileManual(H5File):
    """
    You can manually map datatype attributes to h5file fields
    """
    def __init__(self, path):
        super(FooFileManual, self).__init__(path)
        self.array_float = DataSet(FooDatatype.array_float, self)
        self.array_int = DataSet(FooDatatype.array_int, self)
        self.scalar_int = Scalar(FooDatatype.scalar_int, self)
        self.abaz = Reference(FooDatatype.abaz, self)


class FooFile(H5File):
    # you can map fewer fields by explicitly listing the mapped fields
    trait = FooDatatype
    fields = [
        FooDatatype.array_float,
        FooDatatype.array_int,
        FooDatatype.scalar_int,
        # FooDatatype.some_transient omitted
        FooDatatype.abaz
    ]


class BarFile(FooFile):
    # inheritance is flattened in the same file
    trait = BarDatatype
    fields = [
        BarDatatype.array_str
    ]



def test_accessors_created_for_all_declarative_attributes(tmph5factory):
    f = BazFile(tmph5factory())
    assert set(BazDataType.declarative_attrs) <= set(f.__dict__)


def test_simple_store_load(tmph5factory, bazFactory):
    baz = BazDataType(miu=numpy.array([0.0, 1.0, 2.0]), scalar_str='topol')
    f = BazFile(tmph5factory())
    f.store(baz)
    f.close()

    ret = BazDataType()
    assert ret.scalar_str is None
    f.load_into(ret)
    assert ret.scalar_str == 'topol'
    assert numpy.all(ret.miu == [0.0, 1.0, 2.0])


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
    with BarFile(tmph5factory()) as barfile:
        barfile.store(bar)
        ret = BarDatatype()
        barfile.load_into(ret)
        assert ret.scalar_int == bar.scalar_int

