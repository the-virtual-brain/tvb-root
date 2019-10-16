import os
import numpy
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tvb.core.neotraits.db import Base
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


@pytest.fixture
def tmph5factory(tmpdir):
    def build(pth='tmp.h5'):
        path = os.path.join(str(tmpdir), pth)
        if os.path.exists(path):
            os.remove(path)
        return path
    return build


@pytest.fixture(scope='session')
def engine(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('tmp')
    path = os.path.join(str(tmpdir), 'tmp.sqlite')
    sqlite_conn_string = r'sqlite:///' + path
    # postgres_conn_string = 'postgresql+psycopg2://tvb:tvb23@localhost:5432/tvb'

    return create_engine(sqlite_conn_string)


@pytest.fixture
def session(engine):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()
