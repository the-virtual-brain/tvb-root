import numpy
from .data import FooDatatype, BarDatatype, BazDataType
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

