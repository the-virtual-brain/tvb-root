# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

from sqlalchemy import String, ForeignKey, Column, Integer, Float
from sqlalchemy.orm import relationship
from tvb.core.neotraits.db import HasTraitsIndex, ensure_float, ensure_int
from tvb.tests.framework.core.neotraits.data import FooDatatype


class BazIndex(HasTraitsIndex):
    # you have to define your primary key, not inherit the one from HasTraitsIndex
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    miu_min = Column(Float)
    miu_max = Column(Float)
    miu_mean = Column(Float)

    scalar_str = Column(String)

    def fill_from_has_traits(self, datatype):
        self.miu_min = ensure_float(datatype.miu.min())
        self.miu_max = ensure_float(datatype.miu.max())
        self.miu_mean = datatype.miu.mean()
        self.scalar_str = datatype.scalar_str


def test_hastraitsindex_sets_tablename_and_polymorphic_identity():
    assert BazIndex.__tablename__ == 'BazIndex'
    assert BazIndex.__mapper_args__['polymorphic_identity'] == 'BazIndex'


class FooIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    array_float_min = Column(Float)
    array_float_max = Column(Float)

    array_int_max = Column(Integer)
    array_int_min = Column(Integer)
    array_int_mean = Column(Float)
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
        self.array_float_min = ensure_float(datatype.array_float.min())
        self.array_float_max = ensure_float(datatype.array_float.max())
        self.array_int_max = ensure_int(datatype.array_int.max())
        self.array_int_min = ensure_int(datatype.array_int.min())
        self.array_int_mean = datatype.array_int.mean()
        self.scalar_int = datatype.scalar_int


class BarIndex(FooIndex):
    # You have to explicitly create a foreign key for the parent FooIndex
    # The polymorphic_on discriminator is set up by the HasTraitsIndex superclass
    id = Column(Integer, ForeignKey(FooIndex.id), primary_key=True)

    array_str_length = Column(Integer)

    def fill_from_has_traits(self, datatype):
        super(BarIndex, self).fill_from_has_traits(datatype)
        self.array_str_length = datatype.array_str.size


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
    assert res[0].miu_min == 0.0
    assert res[0].miu_max == 2.0
    assert res[0].miu_mean == 1.0
    assert res[0].scalar_str == 'tick'


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
    assert res[0].array_str_length == 3
    # inherited field
    assert res[0].array_float_max == 42
    # relationship in the parent class
    assert res[0].abaz.scalar_str == 'tick'
