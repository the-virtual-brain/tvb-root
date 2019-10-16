import numpy
import pytest
from tvb.tests.framework.core.neotraits.data import FooDatatype


@pytest.fixture()
def datatypeinstance():
    return FooDatatype(
        array_float=numpy.linspace(0, 42, 100),
        array_int=numpy.arange(8*8).reshape((8, 8)),
        scalar_str='ana',
        scalar_int=42
    )