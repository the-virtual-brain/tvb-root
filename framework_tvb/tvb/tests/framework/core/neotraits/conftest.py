import numpy
import pytest
from tvb.tests.framework.core.neotraits.data import FooDatatype, BooDatatype


@pytest.fixture()
def fooFactory():
    def build():
        return FooDatatype(
            array_float=numpy.linspace(0, 42, 100),
            array_int=numpy.arange(8*8).reshape((8, 8)),
            scalar_str='ana',
            scalar_int=42
        )
    return build


@pytest.fixture()
def booFactory():
    def build():
        return BooDatatype(
            array_float=numpy.linspace(0, 42, 100),
            array_int=numpy.arange(8*8).reshape((8, 8)),
            scalar_str='ana',
            scalar_int=42,
            array_str=numpy.array(['ana', 'are', 'mere'])
        )
    return build
