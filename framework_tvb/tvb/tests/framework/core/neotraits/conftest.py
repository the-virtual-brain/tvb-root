import numpy
import pytest
from tvb.tests.framework.core.neotraits.data import FooDatatype, BarDatatype, BazDataType


@pytest.fixture()
def bazFactory():
    def build():
        return BazDataType(miu=numpy.array([0.0, 1.0, 2.0]), scalar_str='the baz')
    return build


@pytest.fixture()
def fooFactory():
    def build():
        return FooDatatype(
            array_float=numpy.linspace(0, 42, 100),
            array_int=numpy.arange(8*8).reshape((8, 8)),
            scalar_int=42,
            abaz=BazDataType(miu=numpy.zeros((2,2)), scalar_str='a baz')
        )
    return build


@pytest.fixture()
def barFactory():
    def build():
        return BarDatatype(
            array_float=numpy.linspace(0, 42, 100),
            array_int=numpy.arange(8*8).reshape((8, 8)),
            scalar_int=42,
            array_str=numpy.array(['ana', 'are', 'mere']),
            abaz=BazDataType(miu=numpy.zeros((2,2)), scalar_str='a baz')
        )
    return build


