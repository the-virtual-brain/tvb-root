# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
This private module implements the neotraits declarative machinery.
The basic Attribute Property and their automatic discovery by the Metaclass.
"""
import abc
import inspect
import typing
from tvb.basic.neotraits.ex import TraitTypeError, TraitAttributeError
from tvb.basic.neotraits.info import auto_docstring
from tvb.basic.logger.builder import get_logger

# a logger for the whole traits system
log = get_logger('tvb.traits')


class _Attr(object):
    """
    A private base class of Attributes
    Contains the minimum expected functionality by the declarative system.
    """
    def __init__(self):
        self.field_name = None  # type: typing.Optional[str]  # to be set by metaclass
        self.owner = None       # type: typing.Optional[MetaType]  # to be set by metaclass

    def _assert_have_field_name(self):
        """ check that the fields we expect the be set by metaclass have been set """
        if self.field_name is None:
            # this is the case if the descriptor is not in a class of type MetaType
            raise AttributeError("Declarative attributes can only be declared in subclasses of HasTraits")

    # subclass api

    def _post_bind_validate(self):
        # type: () -> None
        """
        Validates this instance of Attr.
        This is called just after field_name is set, by MetaType.
        We do checks here and not in init in order to give better error messages.
        Attr should be considered initialized only after this has run
        """

    # descriptor protocol

    def __get__(self, instance, owner):
        # type: (typing.Optional['HasTraits'], 'MetaType') -> typing.Any
        self._assert_have_field_name()
        if instance is None:
            # called from class, not an instance
            return self
        # data is stored on the instance in a field with the same name
        return instance.__dict__[self.field_name]


    def __set__(self, instance, value):
        # type: ('HasTraits', typing.Any) -> None
        self._assert_have_field_name()
        instance.__dict__[self.field_name] = value


    def __delete__(self, instance):
        raise TraitAttributeError("can't be deleted", attr=self)

    # A modest attempt of making Attr immutable

    def __setattr__(self, key, value):
        """ After owner is set disallow any field assignment """
        if getattr(self, 'owner', None) is not None:
            raise TraitAttributeError(
                "Can't change an Attr after it has been bound to a class."
                "Reusing Attr instances for different fields is not supported."
            )
        super(_Attr, self).__setattr__(key, value)


    def __delattr__(self, item):
        raise TraitAttributeError("Deleting an Attr field is not supported.")



class _Property(object):
    def __init__(self, fget, attr, fset=None):
        # type: (typing.Callable, _Attr, typing.Optional[typing.Callable]) -> None
        self.fget = fget
        self.fset = fset
        self.__doc__ = fget.__doc__
        self.attr = attr



class MetaType(abc.ABCMeta):
    """
    Metaclass for the declarative traits.
    We inherit ABCMeta so that the users may use @abstractmethod without having to
    deal with 2 meta-classes.
    Even though we do this we don't support the dynamic registration of subtypes to these abc's
    """

    # This is a python metaclass.
    # For an introduction see https://docs.python.org/2/reference/datamodel.html

    # here to avoid some hasattr; is None etc checks. And to make pycharm happy
    # should be harmless and shadowed by _declarative_attrs on the returned classes
    _own_declarative_attrs = ()  # type: typing.Tuple[str, ...] # name of all declarative fields on this class
    _own_declarative_props = ()  # type: typing.Tuple[str, ...]

    # A record of all the classes we have created.
    # note: As this holds references and not weakrefs it will prevent class garbage collection.
    #       Deleting classes would break get_known_subclasses and this cache
    __classes = {}  # type: typing.Dict[str, type]


    def get_known_subclasses(cls, include_abstract=False, include_itself=False):
        # type: (bool, bool) -> typing.Dict[str, typing.Type[MetaType]]
        """
        Returns all subclasses that exist *now*.
        New subclasses can be created after this call,
        after importing a new module or dynamically creating subclasses.
        Use with care. Use after most relevant modules have been imported.
        """
        ret = {}

        for k, c in cls.__classes.items():
            if issubclass(c, cls):
                if inspect.isabstract(c) and not include_abstract:
                    continue
                if c == cls and not include_itself:
                    continue
                ret.update({k: c})
        return ret

    def __walk_mro_inherit_declarations(cls, declaration):
        ret = []
        for super_cls in cls.mro():
            if isinstance(super_cls, MetaType):
                for attr_name in getattr(super_cls, declaration):
                    if attr_name not in ret:  # attr was overridden, don't duplicate
                        ret.append(attr_name)
        return tuple(ret)

    @property
    def declarative_attrs(cls):
        # type: () -> typing.Tuple[str, ...]
        """
        Gathers all the declared attributes, including the ones declared in superclasses.
        This is a meta-property common to all classes with this metatype
        """
        # We walk the mro here. This is in contrast with _own_declarative_attrs which is
        # not computed but cached by the metaclass on the class.
        # Caching is faster, but comes with the cost of taking care of the caches validity
        return cls.__walk_mro_inherit_declarations('_own_declarative_attrs')

    @property
    def declarative_props(cls):
        # type: () -> typing.Tuple[str, ...]
        """
        Gathers all the declared props, including the ones declared in superclasses.
        This is a meta-property common to all classes with this metatype
        """
        return cls.__walk_mro_inherit_declarations('_own_declarative_props')

    # here only to have a similar invocation like declarative_attrs
    # namely type(traited_instance).own_declarative_attrs
    # consider the traited_instance._own_declarative_attrs discouraged
    @property
    def own_declarative_attrs(cls):
        return cls._own_declarative_attrs


    def __new__(mcs, type_name, bases, namespace):
        """
        Gathers the names of all declarative fields.
        Tell each Attr of the name of the field it is bound to.
        """
        # gather all declarative attributes defined in the class to be constructed.
        # validate all declarations before constructing the new type
        attrs = []
        props = []

        for k, v in namespace.items():
            if isinstance(v, _Attr):
                attrs.append(k)
            elif isinstance(v, _Property):
                props.append(k)

        # record the names of the declarative attrs in the _own_declarative_attrs field
        if '_own_declarative_attrs' in namespace:
            raise TraitTypeError('class attribute _own_declarative_attrs is reserved in traited classes')
        if '_own_declarative_props' in namespace:
            raise TraitTypeError('class attribute _own_declarative_props is reserved in traited classes')

        namespace['_own_declarative_attrs'] = tuple(attrs)
        namespace['_own_declarative_props'] = tuple(props)
        # construct the class
        cls = super(MetaType, mcs).__new__(mcs, type_name, bases, namespace)

        # inform the Attr instances about the class their are bound to
        for attr_name in attrs:
            v = namespace[attr_name]
            v.field_name = attr_name
            v.owner = cls
            # noinspection PyProtectedMember
            v._post_bind_validate()

        # do the same for props.
        for prop_name in props:
            v = namespace[prop_name].attr
            v.field_name = prop_name
            v.owner = cls
            # noinspection PyProtectedMember
            v._post_bind_validate()

        # update docstring. Note that this is only possible if cls was created by a metatype in python
        setattr(cls, '__doc__', auto_docstring(cls))

        # update the HasTraits class registry
        mcs.__classes.update({cls.__name__: cls})
        return cls


    # note: Any methods defined here are metamethods, visible from all classes with this metatype
    # AClass.field if not found on AClass will be looked up on the metaclass
    # If you just want a regular private method here name it def __lala(cls)
    # double __ not for any privacy but to reduce namespace pollution
    # double __ will mangle the names and make such lookups fail

    # warn about dynamic Attributes


    def __setattr__(self, key, value):
        """
        Complain if TraitedClass.a = Attr()
        Traits assumes that all attributes are statically declared in the class body
        """
        if isinstance(value, _Attr):
            log.warning('dynamically assigned Attributes are not supported')
        super(MetaType, self).__setattr__(key, value)


    def __delattr__(self, item):
        if isinstance(getattr(self, item, None), _Attr):
            log.warning('Dynamically removing Attributes is not supported')
        super(MetaType, self).__delattr__(item)
