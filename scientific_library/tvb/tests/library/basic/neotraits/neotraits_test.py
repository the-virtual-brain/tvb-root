# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import abc
import uuid
import numpy
import numpy as np
import pytest
from tvb.basic.neotraits._core import TraitProperty
from tvb.basic.neotraits.api import (
    HasTraits, Attr, NArray, Final, List, trait_property,
    Int, Float, Range, cached_trait_property, LinspaceRange, Dim
)
from tvb.basic.neotraits.ex import TraitTypeError, TraitValueError, TraitAttributeError, TraitError


def test_simple_declaration():
    class A(HasTraits):
        fi = Attr(int)


def test_simple_instantiation():
    class A(HasTraits):
        fi = Attr(int)

    a2 = A(fi=12)

    with pytest.raises(TraitError):
        a3 = A(fi_typo=43)


def test_types_enforced():
    class A(HasTraits):
        fi = Attr(int, default=0)

    ai = A()

    with pytest.raises(TypeError):
        ai.fi = 'ana'


def test_defaults_and_required():
    class A(HasTraits):
        fi = Attr(int, default=3)
        ra = Attr(str)
        vi = Attr(str, required=False)

    aok = A(ra='opaline')
    aok.vi = None

    aincomplete = A()

    with pytest.raises(ValueError):
        a = A(ra='opaline')
        a.ra = None

    with pytest.raises(TraitAttributeError):
        aincomplete.ra

    with pytest.raises(TraitAttributeError):
        aincomplete.validate()

    # the default configure calls validate
    with pytest.raises(TraitAttributeError):
        aincomplete.configure()


def test_default_compatible_with_field_type_and_choices():
    with pytest.raises(TraitError):
        class A(HasTraits):
            fi = Attr(int, choices=(3, 4), default=2345)

    with pytest.raises(TraitError):
        class A(HasTraits):
            fi = Attr(int, default=23.45)

    with pytest.raises(TraitError):
        class A(HasTraits):
            fi = Attr(int, choices=(1.2, 4.2))

        a = A()
        a.fi = 1.2  # the error is late, but better than never


def test_postinitchecks():
    class A(HasTraits):
        is_foo = Attr(bool, default=False, doc='is it a foo?')
        name = Attr(str)

        def __init__(self, **kwargs):
            super(A, self).__init__(**kwargs)
            self.name = 'opaline'

    aok = A()


def test_composition():
    class A(HasTraits):
        fi = Attr(int, default=3)
        ra = Attr(str, default='ela')

    class B(A):
        opt = Attr(str, required=False)

    class Comp(HasTraits):
        a = Attr(A)
        b = Attr(B)

    comp = Comp(a=A(), b=B())


def test_inheritance():
    class A(HasTraits):
        a = Attr(str)

    class B(HasTraits):
        b = Attr(int)
        c = Attr(float)

    # mixin should not support Attr
    class Mixin(object):
        m = Attr(int, default=42)

    class Inherit(Mixin, B, A):
        c = Attr(str)

    # c has been correctly overridden and is of type str not float
    t = Inherit(a="ana", c="are", b=2)

    assert t.c == "are"

    # we don't support Attr in mixins
    # For simplicity of implementation it fails quite late at use time not when Inherit is declared
    with pytest.raises(AttributeError):
        t.m = 2


def test_attr_misuse_fails():
    with pytest.raises(TypeError):
        class A(HasTraits):
            a = Attr('ana')


def test_declarative_attrs():
    class A(HasTraits):
        a = Attr(str)

    class B(HasTraits):
        b = Attr(int)
        c = Attr(float)

    class Inherit(B, A):
        c = Attr(str)

    assert set(Inherit.declarative_attrs) == {'gid', 'c', 'b', 'a'}
    assert Inherit._own_declarative_attrs == ('c',)
    assert Inherit.own_declarative_attrs == ('c',)

    t = Inherit(a="ana", c="are", b=2)

    # Notice that declarative_attrs is not available on the instances only on the class.
    # This is a bit unusual, but typical for metamethods
    with pytest.raises(AttributeError):
        t.declarative_attrs

    assert set(type(t).declarative_attrs) == {'gid', 'c', 'b', 'a'}


def test_mutable_defaults():
    class A(HasTraits):
        mut = Attr(list, default=[3, 4])
        mut2 = Attr(list, default=[42])

        def __init__(self, **kwargs):
            super(A, self).__init__(**kwargs)
            # copy the default in the constructor
            # and avoid the common shared default
            self.mut = self.mut[:]

    aok = A()

    assert aok.mut is not A.mut.default
    assert aok.mut2 is A.mut2.default

    class B(HasTraits):
        # a better option is to use a lambda default
        mut_better = Attr(list, default=lambda: [3, 4])

    bok = B()

    assert bok.mut_better is not B.mut_better.default
    assert aok.mut2 is A.mut2.default


def test_mutable_unassigned_defaults():
    class B(HasTraits):
        mut_shared = Attr(list, default=[3, 4])
        mut_better = Attr(list, default=lambda: [3, 4])

    bok = B()
    bok2 = B()
    bok.mut_better.append(42)
    assert bok.mut_better == [3, 4, 42]
    assert bok2.mut_better == [3, 4]

    bok.mut_shared.append(42)
    assert bok.mut_shared == [3, 4, 42]
    assert bok2.mut_shared == [3, 4, 42]


def test_late_attr_binding_fail():
    class F(HasTraits):
        f = Attr(str)

    F.newdynamic = Attr(str)
    del F.f


def test_mro_fail():
    class A(HasTraits): pass

    class B(A): pass

    with pytest.raises(TypeError) as ex:
        class Inherit(A, B):
            """
            simple mro contradiction. rule 1 Inherit(A, B) => A < B in mro ; rule 2 B subclass A => B < A
            """

        assert 'consistent method resolution' in str(ex.value)


def test_narr_simple():
    class Boo(HasTraits):
        x = NArray(shape=(Dim.any, Dim.any), dtype=np.dtype(np.int))

    boo = Boo(x=np.array([[1, 4]]))
    boo.x = np.array([[1], [2]])


def test_narr_enforcing():
    with pytest.raises(TypeError):
        class Boo(HasTraits):
            x = NArray(dtype=np.dtype(np.int), default=np.eye(2))

    with pytest.raises(ValueError):
        # bad ndim default
        class Boo(HasTraits):
            x = NArray(shape=(Dim.any, Dim.any, Dim.any), default=np.linspace(1, 3, 10))

    with pytest.raises(TypeError):
        # default should be ndarray
        class Boo(HasTraits):
            x = NArray(default=[1, 2, 4])

    class Boo(HasTraits):
        x = NArray(shape=(Dim.any, Dim.any), default=np.eye(2))
        y = NArray(dtype=np.dtype(int), default=np.arange(12), domain=list(range(5)))

    # only the defaults are checked for domain compliance
    boo = Boo(y=np.arange(10))
    boo.x = np.eye(5)
    boo.x = np.eye(2, dtype=float)
    # only the defaults are checked for domain compliance
    boo.y = np.arange(12)


def test_narr_accepts_safely_convertible_values():
    class Boo(HasTraits):
        x = NArray(dtype=np.float64)
        y = NArray(dtype=np.float32)

    boo = Boo()
    # float32 -> float64 is safe
    boo.x = numpy.array([2.0, 4.3], dtype=np.float32)
    # destination dtype is kept, assigned value is converted, maybe copied
    assert boo.x.dtype == np.float64
    # int -> float64 is safe
    boo.x = numpy.arange(5)
    # however int -> float32 is not
    with pytest.raises(TraitError):
        boo.y = numpy.arange(5)


def test_choices():
    class A(HasTraits):
        x = Attr(str, default='ana', choices=('ana', 'are', 'mere'))
        cx = Final('ana')

    a = A()

    with pytest.raises(ValueError):
        a.x = 'hell'


def test_const_attr():
    class A(HasTraits):
        cx = Final('ana')
        final_c = Attr(int, final=True)

    a = A()
    # in neotraits uninitialized required attrs with no defaults raise on read
    with pytest.raises(TraitAttributeError):
        a.final_c

    a.final_c = 42

    with pytest.raises(TraitAttributeError):
        a.cx = 'hell'

    with pytest.raises(TraitAttributeError):
        a.final_c = 'hell'

    # revealing the insides.
    a.__dict__['final_c'] = None
    a.final_c = 24


def test_named_dimensions():
    class A(HasTraits):
        x = NArray(dim_names=('time', 'space'), required=False)

    assert A.x.ndim == 2
    a = A()
    a.x = np.eye(4)
    assert a.x.shape == (4, 4)
    assert type(a).x.dim_names[0] == 'time'

    with pytest.raises(TraitValueError):
        # ndim dim_names contradiction
        NArray(dim_names=('time', 'space'), shape=(Dim.any, Dim.any, Dim.any, Dim.any))


def test_ndim_enforced():
    class A(HasTraits):
        x = NArray(shape=(Dim.any, Dim.any))

    a = A()
    with pytest.raises(TraitValueError):
        a.x = numpy.arange(4)


def test_dims():
    class A(HasTraits):
        n_nodes = Dim(doc='number of nodes')
        n_sv = Dim()
        w = NArray(shape=(n_nodes, n_nodes))

    class B(A):
        tsr = NArray(shape=(Dim.any, A.n_nodes, A.n_sv))

    a = A(n_sv=2)

    with pytest.raises(TraitAttributeError):
        # dims are required, so reading a non-initialized one fails
        a.n_nodes

    a.n_nodes = 42
    assert a.n_sv == 2

    b = B(n_sv=2)

    with pytest.raises(TraitAttributeError):
        # w references the undefined n_nodes
        b.w = numpy.eye(2)


def test_lists():
    class A(HasTraits):
        picked_dimensions = List(of=str, default=('time', 'space'), choices=('time', 'space', 'measurement'))

    a = A()

    with pytest.raises(ValueError):
        a.picked_dimensions = ['value not in choices']

    # on set we can complain about types and wrong choices
    with pytest.raises(TypeError):
        a.picked_dimensions = [76]

    a.picked_dimensions = ['time', 'space']
    # however there is little we can do against mutation
    a.picked_dimensions.append(76)

    # unless you assign a tuple
    with pytest.raises(AttributeError):
        a.picked_dimensions = ('time', 'space')
        a.picked_dimensions.append(76)


def test_list_default_right_type():
    with pytest.raises(TypeError):
        class A(HasTraits):
            picked_dimensions = List(of=str, default=('time', 42.24))


def test_list_default_must_respect_choices():
    with pytest.raises(ValueError):
        class A(HasTraits):
            picked_dimensions = List(
                of=str,
                default=('time',),
                choices=('a', 'b', 'c')
            )


def test_str_ndarrays():
    class A(HasTraits):
        s = NArray(dtype='U5')

    a = A(s=np.array(['ana', 'a', 'adus', 'mere']))
    # but users will expect python list[str] like behaviour and then this happens
    a.s[0] = 'Georgiana'
    assert 'Georgiana' != a.s[0]
    assert 'Georg' == a.s[0]

    # dtype(str) is dtype('|S0') so it is the most restrictive thus useless
    with pytest.raises(TypeError):
        class A(HasTraits):
            s = NArray(dtype=str, default=np.array(['eli']))
        # fails because the declared type |S0 is different from |S3
        # it is not only different but not compatible


def test_reusing_attribute_instances_fail():
    with pytest.raises(AttributeError):
        class A(HasTraits):
            a, b = [Attr(int)] * 2


def test_changing_Attr_after_bind_fails():
    class A(HasTraits):
        a = Attr(str)

    with pytest.raises(AttributeError):
        A.a.default = 'ana'

    with pytest.raises(AttributeError):
        del A.a.default


def test_declarative_property():
    class A(HasTraits):
        x = NArray(
            label='th x',
            doc='the mighty x'
        )

        def __init__(self, **kwargs):
            super(A, self).__init__(**kwargs)
            self.x3_calls = 0
            self.expensive_once_calls = 0

        def tw(self):
            return self.x * 2

        x2 = TraitProperty(tw, NArray(label='x * 2'))

        @trait_property(NArray(label='3 * x', doc='use the docstring, it is nicer'))
        def x3(self):
            """
            this will be added to the doc of the NArray
            Encouraged place for doc
            """
            self.x3_calls += 1
            return self.x * 3

        @cached_trait_property(NArray(label='expensive'))
        def expensive_once(self):
            self.expensive_once_calls += 1
            d = self.x[:, numpy.newaxis]
            return d + d.T

    a = A()
    a.x = np.arange(12)
    assert set(A.declarative_props) == {'x3', 'x2', 'expensive_once'}
    assert (a.x2 == a.x * 2).all()
    assert (a.x3 == a.x * 3).all()

    a.x3
    assert a.x3_calls == 2
    a.expensive_once
    assert a.expensive_once_calls == 1
    # this does not call expensive_once again
    assert a.expensive_once.shape == (12, 12)
    assert a.expensive_once_calls == 1
    a.expensive_once = numpy.eye(4, dtype=float)
    assert a.expensive_once.shape == (4, 4)
    # invalidate cache by removing the instance state
    del a.expensive_once
    a.expensive_once
    assert a.expensive_once_calls == 2
    assert a.expensive_once.shape == (12, 12)


def test_bool_failure():
    class A(HasTraits):
        does_it_sink = Attr(bool)

    a = A()

    with pytest.raises(TraitTypeError) as e:
        a.does_it_sink = (numpy.zeros(4) == 0).all()

    # bool is not numpy.bool_ !!
    # should we introduce a Bool Attr like we had to do for the Int ??

    # I'd blame numpy if the following assertion would have failed:
    # WHY python? Why must bool be a int, this is not C and even there ...
    assert True + True == 2
    # for more fun philosophy:
    # truth without truth is falsity
    assert True - True == False
    # Half a truth is a falsity
    assert True // 2 == False


def test_int_attribute():
    class A(HasTraits):
        a = Int()
        b = Int(field_type=np.int8)
        c = Int(field_type=np.uint16)

    ainst = A()
    assert ainst.b == 0

    # type is out of bounds but value is within the bounds. So this is ok
    ainst.b = int(42)
    # values are not only checked for compatibility but converted to the declared type
    assert type(ainst.b) == np.int8
    ainst.b = np.int(4)

    with pytest.raises(TypeError):
        # out of bounds for a int8
        ainst.b = 102345

    with pytest.raises(TypeError):
        # floats can't be safely cast to int
        ainst.a = 3.12

    with pytest.raises(TypeError):
        # floats are not ok even when they don't have decimals
        ainst.a = 4.0

    with pytest.raises(TypeError):
        # negative value is not ok in a unsigned int field
        ainst.c = -1

    # signed to unsigned is ok when value fits

    ainst.c = 42

    with pytest.raises(TypeError):
        # incompatible field type
        class B(HasTraits):
            a = Int(field_type=float)

    with pytest.raises(TypeError):
        # incompatible field default
        class B(HasTraits):
            a = Int(default=1.0)


def test_numerics_respect_choices_and_null():
    with pytest.raises(TraitValueError):
        # choices are respected by numeric attributes
        class B(HasTraits):
            a = Int(default=1, choices=(3, 4))

    class B(HasTraits):
        a = Int(default=1, choices=(1, 3, 4))
        odd_nullable_int = Int(required=False)

    with pytest.raises(TraitValueError):
        # choices are respected by numeric attributes
        binst = B()
        binst.a = 43

    binst = B()
    binst.odd_nullable_int = None

    with pytest.raises(TraitValueError):
        binst = B()
        binst.a = None


def test_float_attribute():
    class A(HasTraits):
        a = Float()
        b = Float(field_type=np.float32)
        c = Float(field_type=np.float16)

    ainst = A()
    # int's are ok
    ainst.a = 1
    ainst.a = int(2 ** 31 - 1)
    # larger floats as well if they actually fit
    ainst.c = np.float64(4)
    # they are converted to the declared types
    assert type(ainst.c) == np.float16

    with pytest.raises(TypeError):
        # out of bounds
        ainst.c = 2 ** 30


def test_deleting_a_declared_attribute_not_supported():
    class A(HasTraits):
        a = Attr(str)

    ainst = A()
    ainst.a = 'ana'

    with pytest.raises(AttributeError):
        del ainst.a


def test_dynamic_attributes_behave_statically_and_warn():
    class A(HasTraits):
        a = Attr(str)

    # these are logged warnings not errors. hard to test. here for coverage
    A.b = Attr(int)

    # the rest of the api assumes you don't change the attributes
    # this is the surprising result, no b in the declarative attr list
    assert set(A.declarative_attrs) == {'a', 'gid'}

    # this fails
    with pytest.raises(AttributeError):
        A().b

    class B(HasTraits):
        a = Attr(str)

    del B.a

    # the rest of the api assumes you don't change the attributes
    # this is the surprising result a not deleted from declarative attr list
    assert set(B.declarative_attrs) == {'a', 'gid'}


def test_declarative_properties():
    class A(HasTraits):
        def __init__(self):
            self._foo = None

        @trait_property(Attr(int))
        def xprop(self):
            return 23

        @trait_property(Attr(str))
        def foo(self):
            return self._foo

        @foo.setter
        def foo(self, val):
            self._foo = val

        @trait_property(Attr(int))
        def lyin_prop(self):
            return 'trickster'

    a = A()
    # read
    assert a.xprop == 23

    with pytest.raises(AttributeError):
        # nope read only
        a.xprop = 2

    with pytest.raises(AttributeError):
        # not supported
        del a.xprop

    # read-write

    a.foo = 'ana'
    assert a.foo == 'ana'

    # properties enforce traited types
    with pytest.raises(TypeError):
        a.foo = 42

    # on trying to read the lying prop we get an error
    with pytest.raises(TypeError):
        a.lyin_prop


def test_declarative_props_enforcing_shapes():
    class A(HasTraits):
        n_node = Int()

        def __init__(self, **kwargs):
            super(A, self).__init__(**kwargs)
            self._weights = None

        @trait_property(NArray(shape=(Dim.any, Dim.any)))
        def weights(self):
            return self._weights

        @weights.setter
        def weights(self, val):
            if val.shape != (self.n_node, self.n_node):
                raise TraitValueError
            self._weights = val

    a = A(n_node=4)
    a.weights = numpy.eye(4)

    with pytest.raises(TraitValueError):
        a.weights = numpy.zeros((2, 3))


def test_get_known_subclasses():
    class A(HasTraits):
        @abc.abstractmethod
        def frob(self):
            pass

    class B(A):
        def frob(self):
            pass

    class C(B):
        pass

    assert set(A.get_known_subclasses().values()) == {B, C}
    assert set(A.get_known_subclasses(include_abstract=True, include_itself=True).values()) == {A, B, C}
    assert set(A.get_known_subclasses(include_abstract=True).values()) == {B, C}
    assert set(A.get_known_subclasses(include_itself=True).values()) == {B, C}


def test_summary_info():
    class Z(HasTraits):
        zu = Attr(int)

    class A(HasTraits):
        a = Attr(str, default='ana')
        b = NArray(dtype=int)
        ref = Attr(field_type=Z)

    ainst = A(b=np.arange(3))
    ainst.title = 'the red rose'
    ainst.gid = uuid.UUID(int=0)
    zinst = Z(zu=2)
    zinst.title = 'Z zuzu'
    ainst.ref = zinst
    summary = ainst.summary_info()

    assert summary['Type'] == 'A'
    assert summary['title'] == 'the red rose'
    assert summary['a'] == "'ana'"
    assert summary['b dtype'].startswith('int')
    assert summary['b shape'].startswith('(3')
    assert summary['b [min, median, max]'] == '[0, 1, 2]'
    assert summary['ref'] == 'Z zuzu'


def test_hastraits_str_does_not_crash():
    class A(HasTraits):
        a = Attr(str, default='ana')
        b = NArray(dtype=int)
        pom = 'prun'

        @trait_property(Attr(int))
        def xprop(self):
            return 23

    ainst = A(b=np.arange(3))
    str(ainst)


def test_hastraits_html_repr_does_not_crash():
    class A(HasTraits):
        a = Attr(str, default='ana')
        b = NArray(dtype=int)

    ainst = A(b=np.arange(3))
    ainst._repr_html_()


def test_special_attributes_disallowed():
    with pytest.raises(TypeError):
        class A(HasTraits):
            _own_declarative_attrs = ('a')

    with pytest.raises(TypeError):
        class A(HasTraits):
            _own_declarative_props = ('a')


def test_linspacerange():
    ls = LinspaceRange(0, 10, 5)
    assert 1 in ls

    numpy.testing.assert_allclose(
        ls.to_array(),
        [0.0, 2.5, 5.0, 7.5, 10],
    )
    # test that repr will not crash
    repr(ls)


def test_range():
    ra = Range(0, 10, 5)
    assert 1 in ra

    numpy.testing.assert_allclose(
        ra.to_array(),
        [0, 5]
    )
    repr(ra)


def test_neotraits_ex():
    class A(HasTraits):
        foo = Int(default=1)

    try:
        a = A()
        a.foo = 'ana'
    except TraitTypeError as ex:
        assert ex.attr.field_name == 'foo'
        msg = str(ex)


def test_pedantic_edge_cases_for_coverage():
    class A(HasTraits):
        x = NArray()

    a = A()
    a.tag('subject', 'john doe')
    str(Attr(object))


def access_attr(a):
    a.s = 'has'
    a.arr = numpy.eye(3) + a.arr


@pytest.mark.benchmark
def test_perf_plain(benchmark):
    class Plain(object):
        def __init__(self):
            self.a = None
            self.arr = numpy.eye(3)

    benchmark(access_attr, Plain())


@pytest.mark.benchmark
def test_perf_trait(benchmark):
    class A(HasTraits):
        s = Attr(str, choices=('Ana', 'has', 'some', 'apples'))
        arr = NArray(shape=(Dim.any, Dim.any), default=numpy.eye(3))

    benchmark(access_attr, A())
