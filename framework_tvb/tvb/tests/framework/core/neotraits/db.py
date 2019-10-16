import pytest
import os

from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.orm.attributes import InstrumentedAttribute

from tvb.core.neotraits.db import Base, NArrayIndex, HasTraitsIndex
from sqlalchemy import create_engine, String, ForeignKey, Column, Integer
from tvb.tests.framework.core.neotraits.data import FooDatatype, BarDatatype, BazDataType


@pytest.fixture(scope='session')
def engine(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('tmp')
    path = os.path.join(str(tmpdir), 'tmp.sqlite')
    return create_engine(r'sqlite:///' + path)


@pytest.fixture
def session(engine):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


class BazIndex(HasTraitsIndex):
    # trait is given. This is an automatic mapping
    trait = BazDataType
    # automatic mapping creates columns for scalar and ndarray traited attributes
    # Only the fields are automatically mapped to Column
    # Id's and inheritance foreign keys have to be explicitly created
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)
    # __tablename__ and __polymorphic_identity__ are automatically set to the name of the class


def test_simple_automapping_maps_all_trait_fields():
    for field_name in (BazDataType.scalar_str.field_name, BazDataType.miu.field_name):
        assert isinstance(getattr(BazIndex, field_name, None), InstrumentedAttribute)


def test_automapping_sets_tablename_and_polymorphic_identity():
    assert BazIndex.__tablename__ == 'BazIndex'
    assert BazIndex.__mapper_args__['polymorphic_identity'] == 'BazIndex'


class FooIndexManual(Base):
    # You can retain full control and map things manually
    __tablename__ = 'foo_index_manual'
    id = Column(Integer, primary_key=True)
    gid = Column(String(32), unique=True)

    array_float_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_float = relationship(NArrayIndex, foreign_keys=array_float_id)

    array_int_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_int = relationship(NArrayIndex, foreign_keys=array_int_id)

    scalar_str = Column(String, nullable=False)
    scalar_int = Column(Integer, nullable=False)

    abaz_id = Column(Integer, ForeignKey(BazIndex.id), nullable=not FooDatatype.abaz.required)
    abaz = relationship(BazIndex, foreign_keys=abaz_id)

    # then you might want to implement a method like this if you map a trait
    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_float = NArrayIndex(dtype=str(datatype.array_float.dtype),
                                       ndim=datatype.array_float.ndim,
                                       shape=str(datatype.array_float.shape))
        self.array_int = NArrayIndex(dtype=str(datatype.array_int.dtype),
                                     ndim=datatype.array_int.ndim,
                                     shape=str(datatype.array_int.shape))
        self.scalar_str = datatype.scalar_str
        self.scalar_int = datatype.scalar_int


class PartialFooIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)
    trait = FooDatatype
    # You can map fewer fields than the declared trait attributes
    # by explicitly listing the fields to be mapped
    fields = [
        FooDatatype.array_float,
        FooDatatype.array_int,
        FooDatatype.scalar_int,
        # FooDatatype.some_transient omitted
    ]
    # You can also add additional columns
    baa = Column(String(4), default='baax')
    # If an attribute is another HasTraits it is *NOT* automatically mapped
    # you have to create it
    abaz_id = Column(Integer, ForeignKey(BazIndex.id), nullable=not FooDatatype.abaz.required)
    abaz = relationship(BazIndex, foreign_keys=abaz_id)


def test_automapping_mixed_maps_only_explicitly_declared_fields():
    assert not hasattr(PartialFooIndex, 'some_transient')


class BarIndex(PartialFooIndex):
    # Similar to composition, inheritance is not magically set up for you
    # You have to explicitly create a foreign key for the parent FooIndex
    id = Column(Integer, ForeignKey(PartialFooIndex.id), primary_key=True)
    # Fields of the parent class of BarDatatype are not automatically mapped.
    trait = BarDatatype
    # the fill_from_has_traits however will magically fill the superclass fields


class FlatBarIndex(HasTraitsIndex):
    # Or you can opt not to use any inheritance and map the Bar trait and it's Foo parent to a single index
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)
    trait = BarDatatype
    fields = BarDatatype.declarative_attrs


def test_schema(session):
    session.query(FooIndexManual)


def test_simple_store_load(session, bazFactory):
    baz = bazFactory()
    baz.scalar_str = 'tick'
    bazidx = BazIndex()
    bazidx.fill_from_has_traits(baz)
    session.add(bazidx)
    session.commit()

    res = session.query(BazIndex)
    assert res.count() == 1
    assert res[0].miu.dtype == 'float64'
    assert res[0].scalar_str == 'tick'


def test_aggregate_store_load(session, fooFactory):
    for i in range(2):
        foo = fooFactory()
        foo.abaz.scalar_str = 'baz {}'.format(i)

        bazidx = BazIndex()
        bazidx.fill_from_has_traits(foo.abaz)

        fooidx = PartialFooIndex()
        fooidx.fill_from_has_traits(foo)
        fooidx.scalar_int = i
        fooidx.abaz = bazidx

        session.add(fooidx)

    session.commit()

    res = session.query(PartialFooIndex)
    assert res.count() == 2
    assert res[0].abaz.scalar_str == 'baz 0'
    assert res[1].abaz.scalar_str == 'baz 1'


def test_store_load_inheritance(session, barFactory, bazFactory):
    baz = bazFactory()
    baz.scalar_str = 'tick'
    bazidx = BazIndex()
    bazidx.fill_from_has_traits(baz)
    session.add(bazidx)

    bar = barFactory()
    baridx = BarIndex()
    # this fills all non-relation fields
    baridx.fill_from_has_traits(bar)
    # relationships have to be explicitly filled
    baridx.abaz = bazidx

    session.add(baridx)
    session.commit()

    res = session.query(BarIndex)
    assert res.count() == 1
    # own field
    assert res[0].array_str.dtype == '|S4'
    # inherited field
    assert res[0].array_float.dtype == 'float64'
    # relationsip in the parent class
    assert res[0].abaz.scalar_str == 'tick'


def test_flat_inheritance_map(session, barFactory):
    bar = barFactory()

    flatbaridx = FlatBarIndex()
    flatbaridx.fill_from_has_traits(bar)

    session.add(flatbaridx)
    session.commit()
