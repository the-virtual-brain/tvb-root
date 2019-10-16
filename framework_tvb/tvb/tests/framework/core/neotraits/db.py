import pytest
import os

from sqlalchemy.orm import sessionmaker, relationship
from tvb.core.neotraits import db
from tvb.core.neotraits.db import Base
from sqlalchemy import create_engine, String, ForeignKey, Column, Integer
from tvb.tests.framework.core.neotraits.data import TestDatatype


class TestDatatypeIndexManual(Base):
    __tablename__ = 'testdatatype_manual'
    id = Column(Integer, primary_key=True)

    array_float_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_float = relationship(db.NArray, foreign_keys=array_float_id)

    array_int_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_int = relationship(db.NArray, foreign_keys=array_int_id)

    scalar_str = Column(String, nullable=False)
    scalar_int = Column(Integer, nullable=False)

    def from_datatype(self, datatype):
        self.array_float = db.NArray(dtype=str(datatype.array_float.dtype),
                                     ndim=datatype.array_float.ndim,
                                     shape=str(datatype.array_float.shape))
        self.array_int = db.NArray(dtype=str(datatype.array_int.dtype),
                                   ndim=datatype.array_int.ndim,
                                   shape=str(datatype.array_int.shape))
        self.scalar_str = datatype.scalar_str
        self.scalar_int = datatype.scalar_int


class TestDatatypeIndex(Base):
    trait = TestDatatype
    fields = [
        TestDatatype.array_float,
        TestDatatype.array_int,
        TestDatatype.scalar_int,
        TestDatatype.scalar_str
    ]


@pytest.fixture(scope='session')
def engine():
    tmpdir = r'C:\Users\mihai\cod\tmp'
    path = os.path.join(tmpdir, 'tmp.sqlite')
    return create_engine(r'sqlite:///' + path)


@pytest.fixture
def session(engine):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_schema(session):
    session.query(TestDatatypeIndexManual)


def test_store_load(session, datatypeinstance):
    datatype_index = TestDatatypeIndexManual()
    datatype_index.from_datatype(datatypeinstance)
    session.add_all([datatype_index])
    session.commit()

    res = session.query(TestDatatypeIndexManual)
    assert res.count() == 1
    assert res[0].array_float.dtype == 'float64'
