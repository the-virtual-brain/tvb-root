import inspect
import types

import numpy
import typing
import abc
import logging
import uuid
from .info import trait_object_str, auto_docstring, trait_object_repr_html, narray_summary_info
from .ex import TraitAttributeError, TraitTypeError, TraitValueError

# a logger for the whole traits system
log = logging.getLogger('tvb.traits')

# once in python > 3.6 simplify the name book keeping with __set_name__


class Attr(object):
    """
    An Attr declares the following about the attribute it describes:
    * the type
    * a default value shared by all instances
    * if the value might be missing
    * documentation
    It will resolve to attributes on the instance.
    """

    # This class is a python data descriptor.
    # For an introduction see https://docs.python.org/2/howto/descriptor.html

    def __init__(
        self, field_type, default=None, doc='', label='', required=True, readonly=False, choices=None
    ):
        # type: (type, typing.Any, str, str, bool, bool, typing.Optional[tuple]) -> None
        """
        :param field_type: the python type of this attribute
        :param default: A shared default value. Behaves like class level attribute assignment.
                        Take care with mutable defaults.
        :param doc: Documentation for this field.
        :param label: A short description.
        :param required: required fields should not be None. Not strongly enforced.
        :param readonly: If assignment should be prohibited.
        :param choices: A tuple of the values that this field is allowed to take.
        """
        self.field_name = None  # type: typing.Optional[str]  # to be set by metaclass
        self.owner = None       # type: typing.Optional[MetaType]  # to be set by metaclass
        self.field_type = field_type
        self.default = default
        self.doc = doc
        self.label = label
        self.required = bool(required)
        self.readonly = bool(readonly)
        self.choices = choices

    # subclass api


    def _post_bind_validate(self):
        # type: () -> None
        """
        Validates this instance of Attr.
        This is called just after field_name is set, by MetaType.
        We do checks here and not in init in order to give better error messages.
        Attr should be considered initialized only after this has run
        """
        if not isinstance(self.field_type, type):
            msg = 'Field_type must be a type not {!r}. Did you mean to declare a default?'.format(
                self.field_type
            )
            raise TraitTypeError(msg, attr=self)

        skip_default_checks = self.default is None or isinstance(self.default, types.FunctionType)

        if not skip_default_checks and not isinstance(self.default, self.field_type):
            msg = 'Attribute should have a default of type {} not {}'.format(
                self.field_type, type(self.default)
            )
            raise TraitTypeError(msg, attr=self)

        if self.choices is not None and not skip_default_checks:
            if self.default not in self.choices:
                msg = 'The default {} must be one of the choices {}'.format(self.default, self.choices)
                raise TraitTypeError(msg, attr=self)

        # heuristic check for mutability. might be costly. hasattr(__hash__) is fastest but less reliable
        try:
            hash(self.default)
        except TypeError:
            log.warning('Field seems mutable and has a default value. '
                        'Consider using a lambda as a value factory \n   attribute {}'.format(self))
        # we do not check here if we have a value for a required field
        # it is too early for that, owner.__init__ has not run yet


    def _validate_set(self, instance, value):
        # type: ('HasTraits', typing.Any) -> typing.Any
        """
        Called before updating the value of an attribute.
        It checks the type *AND* returns the valid value.
        You can override this for further checks. Call super to retain this check.
        Raise if checks fail.
        You should return a cleaned up value if validation passes
        """
        if value is None:
            if self.required:
                raise TraitValueError("Attribute is required. Can't set to None", attr=self)
            else:
                return value

        if not isinstance(value, self.field_type):
            raise TraitTypeError("Attribute can't be set to an instance of {}".format(type(value)), attr=self)
        if self.choices is not None:
            if value not in self.choices:
                raise TraitValueError("Value {!r} must be one of {}".format(value, self.choices), attr=self)
        return value


    def _assert_have_field_name(self):
        """ check that the fields we expect the be set by metaclass have been set """
        if self.field_name is None:
            # this is the case if the descriptor is not in a class of type MetaType
            raise AttributeError("Declarative attributes can only be declared in subclasses of HasTraits")

    # descriptor protocol

    def __get__(self, instance, owner):
        # type: (typing.Optional['HasTraits'], 'MetaType') -> typing.Any
        self._assert_have_field_name()
        if instance is None:
            # called from class, not an instance
            return self
        # data is stored on the instance in a field with the same name
        # If field is not on the instance yet, return the class level default
        # (this attr instance is a class field, so the default is for the class)
        # This is consistent with how class fields work before they are assigned and become instance bound
        if self.field_name not in instance.__dict__:
            if isinstance(self.default, types.FunctionType):
                default = self.default()
            else:
                default = self.default

            # Unless we store the default on the instance, this will keep returning self.default()
            # when the default is a function. So if the default is mutable, any changes to it are
            # lost as a new one is created every time.
            instance.__dict__[self.field_name] = default

        return instance.__dict__[self.field_name]


    def __set__(self, instance, value):
        # type: ('HasTraits', typing.Any) -> None
        self._assert_have_field_name()
        if self.readonly:
            raise TraitAttributeError("can't set readonly attribute")
        value = self._validate_set(instance, value)

        instance.__dict__[self.field_name] = value


    def __delete__(self, instance):
        raise TraitAttributeError("can't be deleted", attr=self)


    def _defined_on_str_helper(self):
        if self.owner is not None:
            return '{}.{}.{} = {}'.format(
                self.owner.__module__,
                self.owner.__name__,
                self.field_name,
                type(self).__name__
            )
        else:
            return '{}'.format(type(self).__name__)


    def __str__(self):
        return '{}(field_type={}, default={!r}, required={})'.format(
            self._defined_on_str_helper(), self.field_type, self.default, self.required
        )

    # A modest attempt of making Attr immutable

    def __setattr__(self, key, value):
        """ After owner is set disallow any field assignment """
        if getattr(self, 'owner', None) is not None:
            raise TraitAttributeError(
                "Can't change an Attr after it has been bound to a class."
                "Reusing Attr instances for different fields is not supported."
            )
        super(Attr, self).__setattr__(key, value)


    def __delattr__(self, item):
        raise TraitAttributeError("Deleting an Attr field is not supported.")


class _Property(object):
    def __init__(self, fget, attr, fset=None):
        # type: (typing.Callable, Attr, typing.Optional[typing.Callable]) -> None
        self.fget = fget
        self.fset = fset
        self.__doc__ = fget.__doc__
        self.attr = attr


class CachedTraitProperty(_Property):
    # This is a *non-data* descriptor
    # Once a field with the same name exists on the instance it will
    # take precedence before this non-data descriptor
    # This means that after the first __get__ which sets a same-name instance attribute
    # this will not be called again. Thus this is a cache.
    # To refresh the cache one could delete the instance attr.

    def __get__(self, instance, owner):
        # type: (typing.Optional['HasTraits'], 'MetaType') -> typing.Any
        if instance is None:
            return self
        ret = self.fget(instance)
        # mhtodo the error messages generated by this will be confusing
        # noinspection PyProtectedMember
        ret = self.attr._validate_set(instance, ret)
        # set the instance same-named attribute which becomes the cache
        setattr(instance, self.attr.field_name, ret)
        return ret



class TraitProperty(_Property):

    def __get__(self, instance, owner):
        # type: (typing.Optional['HasTraits'], 'MetaType') -> typing.Any
        if instance is None:
            return self
        ret = self.fget(instance)
        # mhtodo the error messages generated by this will be confusing
        # noinspection PyProtectedMember
        ret = self.attr._validate_set(instance, ret)
        return ret

    def setter(self, fset):
        # return a copy of self that has fset. It will overwrite the current one in the
        # owning class as the attributes have the same name and the setter comes after the getter
        return type(self)(self.fget, self.attr, fset)

    def __set__(self, instance, value):
        # type: ('HasTraits', typing.Any) -> None
        if self.fset is None:
            raise TraitAttributeError("Can't set attribute. Property is read only. In " + str(self))
        # mhtodo the error messages generated by this will be confusing
        # noinspection PyProtectedMember
        value = self.attr._validate_set(instance, value)
        self.fset(instance, value)

    def __delete__(self, instance):
        raise TraitAttributeError("can't delete a traitproperty")

    def __str__(self):
        return 'TraitProperty(attr={}, fget={}'.format(self.attr, self.fget)



def trait_property(attr):
    # type: (Attr) -> typing.Callable[[typing.Callable], TraitProperty]
    """
    A read only property that has a declarative attribute associated with.
    :param attr: the declarative attribute that describes this property
    """
    if not isinstance(attr, Attr):
        raise TypeError('@trait_property(attr) attribute argument required.')

    def deco(func):
        return TraitProperty(func, attr)
    return deco


def cached_trait_property(attr):
    # type: (Attr) -> typing.Callable[[typing.Callable], CachedTraitProperty]
    """
    A lazy evaluated attribute.
    Transforms the decorated method into a cached property.
    The method will be called once to compute a value.
    The value will be stored in an instance attribute with
    the same name as the decorated function.
    :param attr: the declarative attribute that describes this property
    """
    if not isinstance(attr, Attr):
        raise TypeError('@cached_trait_property(attr) attribute argument required.')

    def deco(func):
        return CachedTraitProperty(func, attr)
    return deco


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
    __classes = []  # type: typing.List[type]


    def get_known_subclasses(cls, include_abstract=False):
        # type: (bool) -> typing.Tuple[typing.Type[MetaType], ...]
        """
        Returns all subclasses that exist *now*.
        New subclasses can be created after this call,
        after importing a new module or dynamically creating subclasses.
        Use with care. Use after most relevant modules have been imported.
        """
        ret = []

        for c in cls.__classes:
            if issubclass(c, cls):
                if inspect.isabstract(c) and not include_abstract:
                    continue
                ret.append(c)
        return tuple(ret)

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

        for k, v in namespace.iteritems():
            if isinstance(v, Attr):
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
        mcs.__classes.append(cls)
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
        if isinstance(value, Attr):
            log.warning('dynamically assigned Attributes are not supported')
        super(MetaType, self).__setattr__(key, value)


    def __delattr__(self, item):
        if isinstance(getattr(self, item, None), Attr):
            log.warning('Dynamically removing Attributes is not supported')
        super(MetaType, self).__delattr__(item)



class HasTraits(object):
    __metaclass__ = MetaType

    # The base __init__ and __str__ rely upon metadata gathered by MetaType
    # we could have injected these in MetaType, but we don't need meta powers
    # this is simpler to grok

    gid = Attr(field_type=uuid.UUID)

    def __init__(self, **kwargs):
        """
        The default init accepts kwargs for all declarative attrs
        and sets them to the given values
        """
        # cls just to emphasise that the metadata is on the class not on instances
        cls = type(self)

        # defined before the kwargs loop, so that a title or gid Attr can overwrite this defaults

        self.gid = uuid.uuid4()
        """ 
        gid identifies a specific instance of the hastraits
        it is used by serializers as an identifier.
        For non-datatype HasTraits this is less usefull but still
        provides a unique id for example for a model configuration
        """  # these strings are interpreted as docstrings by many tools, not by python though

        self.title = '{} gid: {}'.format(self.__class__.__name__, self.gid)
        """ a generic name that the user can set to easily recognize the instance """

        for k, v in kwargs.iteritems():
            if k not in cls.declarative_attrs:
                raise TraitTypeError(
                    'Valid kwargs for type {!r} are: {}. You have given: {!r}'.format(
                        cls, repr(cls.declarative_attrs), k
                    )
                )
            setattr(self, k, v)

        self.tags = {}
        """
        a generic collections of tags. The trait system is not using them
        nor should any other code. They should not alter behaviour
        They should describe the instance for the user
        """


    def __str__(self):
        return trait_object_str(self)


    def _repr_html_(self):
        return trait_object_repr_html(self)

    def tag(self, tag_name, tag_value=None):
        # type: (str, str) -> None
        """
        Add a tag to this trait instance.
        The tags are for user to recognize and categorize the instances
        They should never influence the behaviour of the program
        :param tag_name: an arbitrary tag
        :param tag_value: an optional tag value
        """
        self.tags[str(tag_name)] = str(tag_value)


    def validate(self):
        """
        Check that the internal invariants of this class are satisfied.
        Not meant to ensure that that is the case.
        Use configure for that.
        The default configure calls this before it returns.
        It complains about missing required attrs
        Can be overridden in subclasses
        """
        cls = type(self)

        for k in cls.declarative_attrs:
            # these getattr's call the descriptors, should we bypass them?
            attr = getattr(cls, k)
            if attr.required and getattr(self, k) is None:
                # log.warning(
                raise TraitValueError(
                    'Attribute is required. You should set it or declare a default',
                    attr=attr
                )


    def configure(self, *args, **kwargs):
        """
        Ensures that invariant of the class are satisfied.
        Override to compute uninitialized state of the class.
        """
        self.validate()


    def summary_info(self):
        # type: () -> typing.Dict[str, str]
        """
        A more structured __str__
        A 2 column table represented as a dict of str->str
        The default __str__ and html representations of this object are derived from
        this table.
        Override this method and return such a table filled with instance information
        that informs the user about your instance
        """
        cls = type(self)
        ret = {'Type': cls.__name__}
        if self.title:
            ret['title'] = str(self.title)

        for aname in cls.declarative_attrs:
            attr_field = getattr(self, aname)
            if isinstance(attr_field, numpy.ndarray):
                ret.update(narray_summary_info(attr_field, ar_name=aname))
            elif isinstance(attr_field, HasTraits):
                ret[aname] = attr_field.title
            else:
                ret[aname] = repr(attr_field)
        return ret
