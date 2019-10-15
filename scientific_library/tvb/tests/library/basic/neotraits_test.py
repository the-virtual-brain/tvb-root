import abc

import numpy
import numpy as np
import pytest

from tvb.basic.neotraits._attr import LinspaceRange
from tvb.basic.neotraits._core import TraitProperty
from tvb.basic.neotraits.api import (
    HasTraits, Attr, NArray, Const, List, traitproperty,
    Int, Float, Range
)


def test_simple_declaration():
    class A(HasTraits):
        fi = Attr(int)


def test_simple_instantiation():
    class A(HasTraits):
        fi = Attr(int)

    a2 = A(fi=12)


def test_types_enforced():
    class A(HasTraits):
        fi = Attr(int, default=0)

    ai = A()

    with pytest.raises(TypeError):
        ai.fi = 'ana'


@pytest.mark.skip('required checks not yet strict. feature under review')
def test_defaults_and_required():
    class A(HasTraits):
        fi = Attr(int, default=3)
        ra = Attr(str)
        vi = Attr(str, required=False)

    aok = A(ra='opaline')

    with pytest.raises(ValueError):
        a2 = A()


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

    assert set(Inherit.declarative_attrs) == {'c', 'b', 'a'}
    assert Inherit._own_declarative_attrs == ('c',)
    assert Inherit.own_declarative_attrs == ('c', )

    t = Inherit(a="ana", c="are", b=2)

    # Notice that declarative_attrs is not available on the instances only on the class.
    # This is a bit unusual, but typical for metamethods
    with pytest.raises(AttributeError):
        t.declarative_attrs

    assert type(t).declarative_attrs == ('c', 'b', 'a')



def test_mutable_defaults():
    class A(HasTraits):
        mut = Attr(list, default=[3, 4])
        mut2 = Attr(list, default=[42])

        def __init__(self, **kwargs):
            super(A, self).__init__(**kwargs)
            self.mut = self.mut[:]
    aok = A()

    assert aok.mut is not A.mut.default
    assert aok.mut2 is A.mut2.default


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
        x = NArray(ndim=2, dtype=np.dtype(np.int))

    boo = Boo(x=np.array([[1, 4]]))
    boo.x = np.array([[1], [2]])


def test_narr_enforcing():
    with pytest.raises(ValueError):
        class Boo(HasTraits):
            x = NArray(dtype=np.dtype(np.int), default=np.eye(2))

    with pytest.raises(ValueError):
        class Boo(HasTraits):
            x = NArray(ndim=3, default=np.linspace(1, 3, 10))

    class Boo(HasTraits):
        x = NArray(ndim=2, default=np.eye(2))
        y = NArray(dtype=np.dtype(int), default=np.arange(12), domain=xrange(5))

    # only the defaults are checked for domain compliance
    boo = Boo(y=np.arange(10))
    boo.x = np.eye(5)
    boo.x = np.eye(2, dtype=float)
    # only the defaults are checked for domain compliance
    boo.y = np.arange(12)


def test_choices():
    class A(HasTraits):
        x = Attr(str, default='ana', choices=('ana', 'are', 'mere'))
        cx = Const('ana')

    a = A()

    with pytest.raises(ValueError):
        a.x = 'hell'


def test_named_dimensions():
    class A(HasTraits):
        x = NArray(dim_names=('time', 'space'), required=False)

    assert A.x.ndim == 2
    a = A()
    a.x = np.eye(4)
    assert a.x.shape == (4, 4)
    assert type(a).x.dim_names[0] == 'time'


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
    with pytest.raises(TypeError):
        class A(HasTraits):
            picked_dimensions = List(
                of=str,
                default=('time',),
                choices=('a', 'b', 'c')
            )


def test_str_ndarrays():
    class A(HasTraits):
        s = NArray(dtype='S5')

    a = A(s=np.array(['ana', 'a', 'adus', 'mere']))
    # but users will expect python list[str] like behaviour and then this happens
    a.s[0] = 'Georgiana'
    assert 'Georgiana' != a.s[0]
    assert 'Georg' == a.s[0]

    # dtype(str) is dtype('|S0') so it is the most restrictive thus useless
    with pytest.raises(ValueError):
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

        def tw(self):
            return self.x * 2

        @traitproperty(NArray(label='3 * x', doc='use the docstring, it is nicer'))
        def x3(self):
            """
            this will be added to the doc of the NArray
            Encouraged place for doc
            """
            return self.x * 3

        x2 = TraitProperty(tw, NArray(label='x * 2'))

    a = A()
    a.x = np.arange(12)
    assert set(A.declarative_props) == {'x3', 'x2'}
    assert (a.x2 == a.x * 2).all()
    assert (a.x3 == a.x * 3).all()


def test_int_attribute():
    class A(HasTraits):
        a = Int()
        b = Int(field_type=np.int8)
        c = Int(field_type=np.uint16)

    ainst = A()
    assert ainst.b == 0

    # type is out of bounds but value is within the bounds. So this is ok
    ainst.b = long(42)
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


def test_float_attribute():
    class A(HasTraits):
        a = Float()
        b = Float(field_type=np.float32)
        c = Float(field_type=np.float16)

    ainst = A()
    # int's are ok
    ainst.a = 1
    ainst.a = 2**61
    # larger floats as well if they actually fit
    ainst.c = np.float64(4)
    # they are converted to the declared types
    assert type(ainst.c) == np.float16

    with pytest.raises(TypeError):
        # out of bounds
        ainst.c = 2**30


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

    # this fails
    with pytest.raises(AttributeError):
        A().b

    class B(HasTraits):
        a = Attr(str)

    del B.a

    with pytest.raises(AttributeError):
        B()



def test_declarative_properties_are_readonly():
    class A(HasTraits):
        @traitproperty(Attr(int))
        def xprop(self):
            return 23

    a = A()
    assert a.xprop == 23

    with pytest.raises(AttributeError):
        a.xprop = 2

    with pytest.raises(AttributeError):
        del a.xprop


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

    assert set(A.get_known_subclasses()) == {B, C}
    assert set(A.get_known_subclasses(include_abstract=True)) == {A, B, C}


def test_summary_info():
    class Z(HasTraits):
        zu = Attr(int)

    class A(HasTraits):
        a = Attr(str, default='ana')
        b = NArray(dtype=int)
        ref = Attr(field_type=Z)

    ainst = A(b=np.arange(3))
    ainst.title = 'the red rose'
    zinst = Z(zu=2)
    zinst.title = 'Z zuzu'
    ainst.ref = zinst
    summary = ainst.summary_info()

    assert summary == {
        'Type': 'A',
        'title': 'the red rose',
        'a': "'ana'",
        'b dtype': 'int32',
        'b shape': '(3L,)',
        'b [min, median, max]': '[0, 1, 2]',
        'ref': 'Z zuzu',
    }

def test_hastraits_str_does_not_crash():
    class A(HasTraits):
        a = Attr(str, default='ana')
        b = NArray(dtype=int)
        pom = 'prun'

        @traitproperty(Attr(int))
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

