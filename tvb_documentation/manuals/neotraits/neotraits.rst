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
        eye = NArray(dtype=int, default=numpy.eye(4))
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


Attribute options
-----------------

All attributes can have a default value. This value is a class attribute and is
shared by all instances. For mutable data this might lead to surprising behaviour.
In that case you can use a factory function as a default.

.. code-block:: python

    class A(HasTraits):
        a = Attr(field_type=dict, default=lambda: {'factory': 'default'})

By default attributes are required. Reading an uninitialized one will fail

.. code-block:: python

    class A(HasTraits):
        a = Attr(str)
        b = Attr(str, default='hello')
        c = Attr(str, required=False)

    >>> ainst = A()
    >>> ainst.a
    TraitAttributeError: required attribute referenced before assignment. Use a default or assign a value before reading it
  attribute __main__.A.a = Attr(field_type=<type 'str'>, default=None, required=True)

If a default is present then you get that

.. code-block:: python

    >>> ainst.b
    'hello'

An attribute may be declared as not required.
When you read an optional attribute with no default you get None.

.. code-block:: python

    >>> ainst.c is None
    True


Numpy arrays and dimensions
---------------------------

Narray attributes can declare a symbolic shape.
Note that when attributes are declared no instance is yet created.
So you cannot declare a concrete shape.

We define the shape by referring to other attributes defined in the same class.

.. code-block:: python

    class A(HasTraits):
        n_nodes = Dim(doc='number of nodes')
        n_sv = Dim()
        w = NArray(shape=(n_nodes, n_nodes))

These Dim's have to be initialized once when an instance is created.
Accessing arrays before the dimensions have been initialized is an error.

.. code-block:: python

    # dims are ordinary attributes, you can initialize them in init
    >>> ainst = A(n_sv=2)
    # w references the undefined n_nodes
    >>> ainst.w = numpy.eye(2)
    tvb.basic.neotraits.ex.TraitAttributeError: Narray's shape references undefined dimension <n_nodes>. Set it before accessing this array
      attribute __main__.A.w = NArray(label='', dtype=float64, default=None, dim_names=(), ndim=2, required=True)

Once the dimensions have been set on an instance they enforce the shapes of the
arrays that use them.

.. code-block:: python

    >>> ainst.n_nodes = 42
    >>> ainst.w = numpy.eye(2)
    tvb.basic.neotraits.ex.TraitValueError: Shape mismatch. Expected (42, 42). Given (2L, 2L). Not broadcasting

We don't allow arithmetic with symbolic dimensions.

.. code-block:: python

    class A(HasTraits):
        n_nodes = Dim(doc='number of nodes')
        flat_w = NArray(shape=(n_nodes * n_nodes, ))

    TypeError: unsupported operand type(s) for *: 'Dim' and 'Dim'

Validation
----------

The Attr declarations already do checks that apply to all instances of a class.
For example the type of an Attr is enforced.

But you might need to do per instance specific validation.

Let's say that for the class below, we want to enforce that ``labels[::-1] == reversed_labels``


.. code-block:: python

    class A(HasTraits):
        labels = NArray(dtype='S64')
        reversed_labels = NArray(dtype='S64')

We can override the `validate` method and do the checks in it.
Validate is not automatically called, but users of the DataType are encouraged to
call it once they are done with populating the instance.

.. code-block:: python

    class A(HasTraits):
        labels = NArray(dtype='S64')
        reversed_labels = NArray(dtype='S64')

        def validate(self):
            super(A, self).validate()
            if (self.labels[::-1] != self.reversed_labels).any():
                raise ValueError

    >>> a = A()
    >>> a.labels = numpy.array(['a', 'b'])
    >>> a.reversed_labels = numpy.array(['a', 'b'])
    >>> a.validate()
    ValueError


This late optional validation might not be a good fit.
If you want an error as soon as you assign a bad array then you
can promote reversed_labels from a declarative attribute to a declarative property.

.. code-block:: python

    class A(HasTraits):
        labels = NArray(dtype='S64')

        def __init__(self, **kwargs):
            super(A, self).__init__(**kwargs)
            self._reversed = None

        @trait_property(NArray(dtype='S64'))
        def reversed_labels(self):
            return self._reversed

        @reversed_labels.setter
        def reversed_labels(self, value):
            if (value[::-1] != self.labels).any():
                raise ValueError
            self._reversed = value

        >>> a = A()
        >>> a.labels = numpy.array(['a', 'b'])
        >>> a.reversed_labels = numpy.array(['a', 'b'])
        >>> a.validate()
        ValueError


.. _neotrait_attr_ref:

Attribute reference
-------------------

.. autoclass:: tvb.basic.neotraits.api.Attr
    :members: __init__

.. autoclass:: tvb.basic.neotraits.api.NArray
    :members: __init__

.. autoclass:: tvb.basic.neotraits.api.Final
    :members: __init__

.. autoclass:: tvb.basic.neotraits.api.Dim
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
    :undoc-members:

.. autofunction:: tvb.basic.neotraits.api.trait_property

.. autofunction:: tvb.basic.neotraits.api.cached_trait_property

.. autoclass:: tvb.basic.neotraits.api.Range
    :members:

.. autoclass:: tvb.basic.neotraits.api.LinspaceRange
    :members:

.. autofunction:: tvb.basic.neotraits.api.narray_describe

.. autofunction:: tvb.basic.neotraits.api.narray_summary_info

.. automodule::  tvb.basic.neotraits.ex
    :members:
