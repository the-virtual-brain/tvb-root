import inspect

import numpy
import typing
import abc
import logging
import uuid
from .info import trait_object_str, auto_docstring, trait_object_repr_html, narray_summary_info

# a logger for the whole traits system
log = logging.getLogger('tvb.traits')



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

    def _err_msg(self, msg):
        # type: (str) -> str
        """
        Adds to a error message information about the Attribute where it occured
        """
        return 'attribute {}.{} = {} : {}'.format(self.owner.__name__, self.field_name, self, msg)

    def _post_bind_validate(self):
        # type: () -> None
        """
        Validates this instance of Attr.
        This is called just after field_name is set, by MetaType.
        We do checks here and not in init in order to give better error messages.
        Attr should be considered initialized only after this has run
        """
        if not isinstance(self.field_type, type):
            msg = 'field_type must be a type not {!r}. Did you mean to use it as the default?'.format(
                self.field_type
            )
            raise TypeError(self._err_msg(msg))

        if self.default is not None and not isinstance(self.default, self.field_type):
            msg = 'should have a default of type {} not {}'.format(self.field_type, type(self.default))
            raise TypeError(self._err_msg(msg))

        if self.choices is not None and self.default is not None:
            if self.default not in self.choices:
                msg = 'the default {} must be one of the choices {}'.format(self.default, self.choices)
                raise TypeError(self._err_msg(msg))

        # heuristic check for mutability. might be costly. hasattr(__hash__) is fastest but less reliable
        try:
            hash(self.default)
        except TypeError:
            log.warning(self._err_msg('field seems mutable and has a default value'))
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
                raise ValueError(self._err_msg("is required. Can't set to None"))
            else:
                return value

        if not isinstance(value, self.field_type):
            raise TypeError(self._err_msg("can't be set to an instance of {}".format(type(value))))
        if self.choices is not None:
            if value not in self.choices:
                raise ValueError(self._err_msg("value {} must be one of {}".format(value, self.choices)))
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
            return self.default
        return instance.__dict__[self.field_name]


    def __set__(self, instance, value):
        # type: ('HasTraits', typing.Any) -> None
        self._assert_have_field_name()
        if self.readonly:
            raise AttributeError("can't set readonly attribute")
        value = self._validate_set(instance, value)

        instance.__dict__[self.field_name] = value


    def __delete__(self, instance):
        msg_where = self._err_msg(type(instance).__name__)
        raise AttributeError(msg_where + "can't be deleted")


    def __str__(self):
        return '{}(field_type={}, default={!r}, required={})'.format(
            type(self).__name__, self.field_type, self.default, self.required
        )

    # A modest attempt of making Attr immutable

    def __setattr__(self, key, value):
        """ After owner is set disallow any field assignment """
        if getattr(self, 'owner', None) is not None:
            raise AttributeError(
                "Can't change an Attr after it has been bound to a class."
                "Reusing Attr instances for different fields is not supported."
            )
        super(Attr, self).__setattr__(key, value)


    def __delattr__(self, item):
        raise AttributeError("Deleting an Attr field is not supported.")



class TraitProperty(object):
    def __init__(self, fget, attr):
        # type: (typing.Callable, Attr) -> None
        self.fget = fget
        self.__doc__ = fget.__doc__
        self.attr = attr

    def __get__(self, instance, owner):
        if instance is None:
            return self
        ret = self.fget(instance)
        # mhtodo the error messages generated by this will be confusing
        # noinspection PyProtectedMember
        ret = self.attr._validate_set(instance, ret)
        return ret

    def __set__(self, instance, value):
        raise AttributeError("can't set attribute. traitproperties are read only")

    def __delete__(self, instance):
        raise AttributeError("can't delete attribute. traitproperties are read only")

    def __str__(self):
        return 'TraitProperty(attr={}, fget={}'.format(self.attr, self.fget)



def traitproperty(attr):
    def deco(func):
        return TraitProperty(func, attr)
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
        New subclasses can be created after this call (after importing a new module or dynamically created ones)
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
            elif isinstance(v, TraitProperty):
                props.append(k)

        # record the names of the declarative attrs in the _own_declarative_attrs field
        if '_own_declarative_attrs' in namespace:
            raise TypeError('class attribute _own_declarative_attrs is reserved in traited classes')
        if '_own_declarative_props' in namespace:
            raise TypeError('class attribute _own_declarative_props is reserved in traited classes')

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


    def _post_init_validate_(cls, instance):
        """
        The code below executes after __init__ body but before object call returns
        It it time to complain about missing required attrs
        Can be overridden in subclasses
        """
        for k in cls.declarative_attrs:
            # these getattr's call the descriptors, should we bypass them?
            attr = getattr(cls, k)
            if attr.required and getattr(instance, k) is None:
                log.warning(
                    'attribute {}.{} = {} is required. '
                    'Initialize it in __init__ or declare a default '.format(cls.__name__, k, attr)
                )

    def __call__(cls, *args, **kwargs):
        """
        Checks that __init__ has initialized required fields
        """
        # call __init__
        instance = super(MetaType, cls).__call__(*args, **kwargs)
        cls._post_init_validate_(instance)
        return instance

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

    def __init__(self, **kwargs):
        """
        The default init accepts kwargs for all declarative attrs
        and sets them to the given values
        """
        # .__class__ just to emphasise that the metadata is on the class not on instances
        cls = type(self)
        for k, v in kwargs.iteritems():
            if k not in cls.declarative_attrs:
                raise TypeError(
                    'Valid kwargs for type {!r} are: {}. You have given: {!r}'.format(
                        cls, repr(cls.declarative_attrs), k
                    )
                )
            setattr(self, k, v)

        self.gid = uuid.uuid4()
        """ 
        gid identifies a specific instance of the hastraits
        it is used by serializers as an identifier.
        For non-datatype HasTraits this is less usefull but still
        provides a unique id for example for a model configuration
        """  # these strings are interpreted as docstrings by many tools, not by python though

        self.title = '{} gid: {}'.format(self.__class__.__name__, self.gid)
        """ a generic name that the user can set to easily recognize the instance """

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

    def configure(self, *args, **kwargs):
        """
        This is here only because a lot of code relies on configure calls.
        This is the default do nothing base implementation
        todo: deleteme
        """

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
