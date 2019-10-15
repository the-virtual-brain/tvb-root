import numpy as np
import pytest
from tvb.basic.traits.neotraits import HasTraits, Attr, NArray, Const, List


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

    # test that constructor fails cause of missing default for the b attr declared in a superclass
    with pytest.raises(ValueError):
        t = Inherit(a="ana", c="are")

    # c has been correctly overridden and is of type str not float
    t = Inherit(a="ana", c="are", b=2)

    assert t.c == "are"

    # we don't support Attr in mixins
    # For simplicity of implementation it fails quite late at use time not when Inherit is declared
    with pytest.raises(AttributeError):
        t.m = 2


def test_declarative_attrs():
    class A(HasTraits):
        a = Attr(str)

    class B(HasTraits):
        b = Attr(int)
        c = Attr(float)

    class Inherit(B, A):
        c = Attr(str)

    assert Inherit.declarative_attrs == ('c', 'b', 'a')
    assert Inherit._own_declarative_attrs == ('c',)

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
        a.picked_dimensions = ['times']

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


def test_str_ndarrays_are_problematic():
    class A(HasTraits):
        s = NArray(dtype=str)

    a = A(s=np.array(['ana', 'a', 'adus', 'mere']))
    # all seems well and a.s is indeed a np.issubdtype of str
    # but users will expect python list[str] like behaviour and then this happens
    a.s[0] = 'Georgiana'
    assert 'Georgiana' != a.s[0]
    assert 'Geor' == a.s[0]

    # dtype(str) is dtype('|S0') so it is the most restrictive thus useless
    with pytest.raises(ValueError):
        class A(HasTraits):
            s = NArray(dtype=str, default=np.array(['eli']))
        # fails because the declared type |S0 is different from |S3
        # it is not only different but not compatible
    # so do we eliminate strict dtype checks for strings?
    # do we start with a default like |S64?
    # do we create a new attribute String that has relaxed checks and infers dtype from default if it exists?
    # do we burden the user with giving a precise dtype like |S32?
    # do we discourage ndarray[str] in favor of plain python lists? and let storage deal with string sizes
