from sqlalchemy import String, ForeignKey, Column, Integer
from sqlalchemy.orm import relationship
from tvb.core.neotraits.db import NArrayIndex, HasTraitsIndex
from tvb.tests.framework.core.neotraits.data import FooDatatype


class BazIndex(HasTraitsIndex):
    # you have to define your primary key, not inherit the one from HasTraitsIndex
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)
    # __tablename__ and __polymorphic_identity__ are automatically set to the name of the class
    miu_id = Column(Integer, ForeignKey(NArrayIndex.id), nullable=False)
    miu = relationship(NArrayIndex, foreign_keys=miu_id)
    scalar_str = Column(String)

    def fill_from_has_traits(self, datatype):
        self.miu = NArrayIndex.from_ndarray(datatype.miu)
        self.scalar_str = datatype.scalar_str



def test_hastraitsindex_sets_tablename_and_polymorphic_identity():
    assert BazIndex.__tablename__ == 'BazIndex'
    assert BazIndex.__mapper_args__['polymorphic_identity'] == 'BazIndex'



class FooIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_float_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_float = relationship(NArrayIndex, foreign_keys=array_float_id)

    array_int_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    array_int = relationship(NArrayIndex, foreign_keys=array_int_id)
    # simple scalars
    scalar_int = Column(Integer, nullable=False)

    # map aggregation to a relationship
    abaz_id = Column(Integer, ForeignKey(BazIndex.id), nullable=not FooDatatype.abaz.required)
    abaz = relationship(BazIndex, foreign_keys=abaz_id)

    # FooDatatype.some_transient omitted
    # You can also add additional columns
    baa = Column(String(4), default='baax')

    # then you might want to implement a method like this if you map a trait
    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.array_float = NArrayIndex.from_ndarray(datatype.array_float)
        self.array_int = NArrayIndex.from_ndarray(datatype.array_int)
        self.scalar_int = datatype.scalar_int



class BarIndex(FooIndex):
    # You have to explicitly create a foreign key for the parent FooIndex
    # The polymorphic_on discriminator is set up by the HasTraitsIndex superclass
    id = Column(Integer, ForeignKey(FooIndex.id), primary_key=True)

    array_str_id = Column(Integer, ForeignKey(NArrayIndex.id), nullable=False)
    array_str = relationship(NArrayIndex, foreign_keys=array_str_id)

    def fill_from_has_traits(self, datatype):
        super(BarIndex, self).fill_from_has_traits(datatype)
        self.array_str = NArrayIndex.from_ndarray(datatype.array_str)


def test_schema(session):
    session.query(FooIndex)


def test_simple_store_load(session, bazFactory):
    baz = bazFactory()
    baz.scalar_str = 'tick'
    bazidx = BazIndex()
    bazidx.fill_from_has_traits(baz)
    session.add(bazidx)
    session.commit()

    res = session.query(BazIndex)
    assert res.count() == 1
    assert res[0].miu.dtype_kind == 'f'
    assert res[0].scalar_str == 'tick'
    assert res[0].miu.dtype_str == '<f8'


def test_aggregate_store_load(session, fooFactory):
    for i in range(2):
        foo = fooFactory()
        foo.abaz.scalar_str = 'baz {}'.format(i)

        bazidx = BazIndex()
        bazidx.fill_from_has_traits(foo.abaz)

        fooidx = FooIndex()
        fooidx.fill_from_has_traits(foo)
        fooidx.scalar_int = i
        fooidx.abaz = bazidx

        session.add(fooidx)

    session.commit()

    res = session.query(FooIndex)
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
    assert res[0].array_str.dtype_str == '|S32'
    # inherited field
    assert res[0].array_float.dtype_str == '<f8'
    # relationsip in the parent class
    assert res[0].abaz.scalar_str == 'tick'

