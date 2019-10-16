Neotraits
=========

Neotraits is an API that lets you declare class attributes with
type checking and introspection abilities.

If you have used `sqlalchemy declarative`_, `django models`_ or `marshmallow`_ you will not
find neotraits surprising. It is quite similar to `traitlets`_.

.. _sqlalchemy declarative: https://docs.sqlalchemy.org/en/latest/orm/extensions/declarative/
.. _django models: https://docs.djangoproject.com/en/2.1/topics/db/models/
.. _traitlets: https://traitlets.readthedocs.io/en/stable/
.. _marshmallow: https://marshmallow.readthedocs.io/en/3.0/index.html

Quick example
-------------

You subclass HasTraits and declare some attributes:

.. code-block:: python

    from tvb.basic.neotraits.api import HasTraits, Attr

    class Foo(HasTraits):
        """ Foo is a traited class """
        bar = Attr(str, doc='an iron bar', default='iron')
        baz = Attr(int)


The default ``__init__`` takes kwargs for each declared attribute.

.. code-block:: python

    >>> foo = Foo(baz=42)

For each declared attribute an instance attribute of the same name is created.

.. code-block:: python

    # foo.baz is an instance attribute of foo.
    >>> print foo.baz
    42
    # foo.bar is initialized with the default
    >>> foo.bar
    'iron'

Neotraits will enforce the type of the attribute

.. code-block:: python

    >>> foo.baz = 'oops'
    TraitTypeError: Attribute can't be set to an instance of <type 'str'>
        attribute __main__.Foo.baz = Attr(field_type=<type 'int'>, default=None, required=True)


Attributes
----------

A neotrait attribute is an instance of :py:class:`Attr`
that is defined in a class that derives from :py:class:`HasTraits`.

Attributes have attribute-specific parameters. Here is a more complex example.

.. code-block:: python

    class Foo(HasTraits):
        food = Attr(str, choices=('fish', 'chicken'), required=False)
        eye = NArray(dtype=int, ndim=2, default=numpy.eye(4))
        boat = Float(
            default=42.42,
            label='boaty mcboatface',
            doc='''
            The aquatic vehicle par exellence.
            '''
        )

See the :ref:`neotrait_attr_ref` for all the available attributes and their parameters.

Composing and inheriting
------------------------

Attributes can refer to other traited classes. You simply define an attribute with
the type of the referred class :

.. code-block:: python

    class A(HasTraits):
        fi = Attr(int)

    class B(A):
        ra = Attr(str)
        vi = Float()

    class Comp(HasTraits):
        a = Attr(A)
        b = Attr(B)

    comp = Comp(a=A(), b=B())

    comp.a.fi = 234


Inheritance also works as expected.

.. code-block:: python

    class A(HasTraits):
        a = Attr(str)

    class B(HasTraits):
        b = Attr(int)
        c = Attr(float)

    class Inherit(B, A):
        c = Attr(str)

    # c has been correctly overridden and is of type str not float
    >>> t = Inherit(a="ana", c="are", b=2)
    >>> t.c
    are


Introspection
-------------

Neotraits keeps a registry of all traited classes that have been created.

.. code-block:: python

    >>> HasTraits.get_known_subclasses()
    (<class 'tvb.basic.neotraits._core.HasTraits'>, <class '__main__.Foo'>)

A traited class keeps a tuple of all the declared attributes:

.. code-block:: python

   >>> Foo.own_declarative_attrs
   ('baz', 'bar')
   # this includes inherited declarative attributes
   >>> Foo.declarative_attrs
   ('baz', 'bar', 'gid')

The declarative Attr objects can still be found on the class. And you can get
the declaration details:

.. code-block:: python

    >>> print Foo.bar
    __main__.Foo.bar = Attr(field_type=<type 'str'>, default='iron', required=True)
    >>> Foo.bar.required
    True

With these capabilities one can generically list all int attributes:

.. code-block:: python

    >>> some_traited_type = Foo
    >>> for k in some_traited_type.declarative_attrs:
    ...    attr = getattr(some_traited_type, k)
    ...    if attr.field_type == int:
    ...        print k
    baz

Introspection is used by the default ``__init__``, jupyter pretty
printing and automatic docstring generation:

Introspection can be used by tools to generate serializers, db mappings,
gui's and other capabilities.


.. code-block:: ReST

    >>> print Foo.__doc__

    Traited class [__main__.Foo]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

     Foo is a traited class

    Attributes declared
    """""""""""""""""""
    baz : __main__.Foo.baz = Attr(field_type=<type 'int'>, default=None, required=True)
    bar : __main__.Foo.bar = Attr(field_type=<type 'str'>, default='iron', required=True)
        an iron bar
    gid : tvb.basic.neotraits._core.HasTraits.gid = Attr(field_type=<class 'uuid.UUID'>, default=None, required=True)


Properties
----------

It is useful to have properties that are also annotated with attributes.
They behave like read only python properties but they also retain the declarations
and can be used by introspection.

.. code-block:: python

    class A(HasTraits):
        x = NArray(doc='the mighty x')

        @trait_property(NArray(label='3 * x', doc='use the docstring, it is nicer'))
        def x3(self):
            """
            Encouraged place for doc
            """
            return self.x * 3

    >>> a = A()
    >>> a.x = numpy.arange(4)
    >>> a.x3
    array([0, 3, 6, 9])

The class remembers which properties are trait_property-ies

.. code-block:: python

    >>> A.declarative_props
    ('x3', )
    >>> A.x3.attr.label
    '3 * x'

Cached properties are similar to properties. But the method only gets called once.
The second reference to a cached property will return the cached value.

.. code-block:: python

    class A(HasTraits):

        @cached_trait_property(NArray(label='expensive'))
        def expensive_once(self):
            return numpy.eye(1001)



.. _neotrait_attr_ref:

Attribute reference
-------------------

.. autoclass:: tvb.basic.neotraits.api.Attr
    :members: __init__

.. autoclass:: tvb.basic.neotraits.api.NArray
    :members: __init__

.. autoclass:: tvb.basic.neotraits.api.Const
    :members: __init__

.. autoclass:: tvb.basic.neotraits.api.Int
    :members: __init__

.. autoclass:: tvb.basic.neotraits.api.Float
    :members: __init__

.. autoclass:: tvb.basic.neotraits.api.List
    :members: __init__


Reference
---------

.. autoclass:: tvb.basic.neotraits.api.HasTraits
    :members:

.. autodecorator:: tvb.basic.neotraits.api.trait_property

.. autodecorator:: tvb.basic.neotraits.api.cached_trait_property

.. autoclass:: tvb.basic.neotraits.api.Range
    :members:

.. autoclass:: tvb.basic.neotraits.api.LinspaceRange
    :members:

.. autofunction:: tvb.basic.neotraits.api.narray_describe

.. autofunction:: tvb.basic.neotraits.api.narray_summary_info

.. automodule::  tvb.basic.neotraits.ex
    :members:
