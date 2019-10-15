if __name__ == '__main__':
    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.basic.traits.types_basic import Float
from tvb.basic.traits.core import Type


def test_basic_traits_semantics():
    # Type already has stuff in it, the trait attr
    assert len(Type.trait.inits.kwd) == len(Type.trait.inits.pos) == 0
    # Some properties on traitinfo are ..
    assert Type.trait.bound is False
    assert Type.trait.select_multiple is False


class A(Type):
    pass

exta = A()
exta.whoami = 'the module level A'


class B(Type):
    a = A
    a3 = A()
    aext = exta  # are some copy-ing to avoid this sharing situation?


class C(B):
    a = B



def test_attribute_lookup():
    # attribute is recorded in the trait registry for this class
    assert B.trait['a'].__class__ is A
    # A is also a data descriptor. The class lookup is returning self.
    # Confusingly a is an instance created by metaclass
    # That instance is a descriptor
    assert B.a is B.trait['a']
    assert type(B.a) is A
    assert type(B.a3) is A
    # descriptors are deep copied by meta
    assert B.aext is not exta

    ai = A()
    bi = B()

    # trait attribute is not looked up on the class. The meta creates a instance attribute
    # copying the class stuff in it
    assert bi.trait is not A.trait

    # instance gets attributes derived from the declared ones
    assert {'_a', '_a3', '_aext'}.issubset(set(bi.__dict__))
    # surprisingly their values are None
    assert bi._a is None
    # as a data descriptor B.a resolves the lookup to bi._a
    assert bi.a is None
    # but the trait contains some instances
    assert type(bi.trait['a']) is A
    assert type(bi.trait['a3']) is A
    # it seems that the _a attrs are assumed to be handled by mappedtype ?
    # Lets put something in that _a
    fresha = A()
    fresha.whoami = 'a fresh A'
    bi._a = fresha
    # it gets resolved by descriptor
    assert bi.a is fresha
    # all this if bound is true
    assert B.a.trait.bound is True
    # but we can mutate it
    B.a.trait.bound = False
    # then it resolves it like in the class case
    assert bi.a is B.trait['a']

    # bi.trait['a'] behaves like a default but it is of course not looked up on the class
    assert bi.trait['a'] is not B.trait['a']


def test_simple_traits_subclassing_semantics():
    # the trait attribute is copied by the metaclass, and thus statically looked up,
    # not naturally looked up in the superclass
    assert A.trait is not Type.trait
    assert C.trait is not B.trait
    # as .trait records all traited class members this makes sense
    # maybe a property with delegation to super would have been more pythonic than copying parent attrs

    # however this is a surprise
    assert C.aext is not B.aext

# All this is breaking python lookup conventions


def test_declarative_instances():
    # -------------------------
    # a = B in class C is both a declarative statement of type and a weird instance


    class Tst(Type):
        f = Float(default=4.3)
        g = Float
        a = A()  # this confuses schema with data

    # note that Float is not float, it is a declarative descriptor
    # if we treated Float attrs like Type attrs then i would have looked like this
    #class ConsistentWeird(Type):
    #    f = 4.3
    #    g = int  # automatically initialized to int() , look-ed up in _g
    #    h = float(some traited kwargs)

    ti = Tst()
    ti.a = A()

    assert type(Tst.f) is Float
    assert type(ti.f) is float
    # note the difference between Float and A
    assert type(Tst.a) is A
    assert type(ti.a) is A
    # this difference is due to the trait.wraps semantic
    assert ti._f == 4.3
    assert ti.f == 4.3
    assert ti.g == 0.0


def test_init():

    adescr = A()

    class Tst(B):
        f = Float()
        ade = adescr

        def __init__(self):
            # in init the trait is a class property, visible to all
            assert self.trait is Tst.trait
            assert self.a is None

    t = Tst()
    # after __init__ the instance has its own copies of the trait and of the traited attrs
    assert t.trait is not Tst.trait
    assert t.f is not Tst.f


def test_wraps_semantics():
    class A(Type):
        wraps = int
        h = 'kindof hidden'

    class B(Type):
        aaa = A

    b = B()
    # now this is something
    # on the one hand yes it should be 0 as it is declared as an int attribute
    # on the other hand the A.h is now a subtree that is weirdly in b.trai
    assert b.aaa == 0
    assert b.trait['aaa'].h == 'kindof hidden'
    # what makes it odd and confusing is that the trait IS instance state
    b.trait['aaa'].h = 'bilbo'
    assert B.trait['aaa'].h == 'kindof hidden'


