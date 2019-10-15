"""
A simple traits declarative api
todo: rename this module
todo: document the new system here and put a link to extensive docs
"""
import inspect
import typing
import abc
import logging

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

    def __init__(self, field_type=object, default=None, doc='', label='',
                 required=True, readonly=False, choices=None):
        # type: (type, typing.Any, str, str, bool, bool, typing.Optional[tuple]) -> None
        self.field_name = None  # type: str  # to be set by metaclass
        self.owner = None  # type: type  # to be set by metaclass
        self.field_type = field_type
        self.default = default
        self.doc = doc
        self.label = label
        self.required = required
        self.readonly = readonly
        self.choices = choices


    def _err_msg_where(self, defined_in_type_name):
        # type: (str) -> str
        return 'attribute {}.{} = {} : '.format(
            defined_in_type_name, self.field_name, self)


    def _post_bind_validate(self, defined_in_type_name):
        # type: (str) -> None
        """
        Validates this instance of Attr.
        This is called just after field_name is set, by MetaType.
        We do checks here and not in init in order to give better error messages.
        Attr should be considered initialized only after this has run
        """
        if self.default is not None and not isinstance(self.default, self.field_type):
            msg = 'should have a default of type {} not {}'.format(
                self.field_type, type(self.default))
            raise TypeError(self._err_msg_where(defined_in_type_name) + msg)

        if self.choices is not None and self.default is not None:
            if self.default not in self.choices:
                msg = 'the default {} must be one of the choices {}'.format(
                    self.default, self.choices)
                raise TypeError(self._err_msg_where(defined_in_type_name) + msg)

        # heuristic check for mutability. might be costly. hasattr(__hash__) is fastest but less reliable
        try:
            hash(self.default)
        except TypeError:
            log.warning(self._err_msg_where(defined_in_type_name)
                        + 'field seems mutable and has a default value')
        # we do not check here if we have a value for a required field
        # it is too early for that, owner.__init__ has not run yet


    def _validate_set(self, instance, value):
        """
        Called before updating the value of an attribute.
        It checks the type.
        You can override this for further checks. Call super to retain this check.
        Raise if checks fail.
        """
        if value is None and not self.required:
            return
        if not isinstance(value, self.field_type):
            msg_where = self._err_msg_where(type(instance).__name__)
            raise TypeError(msg_where + "can't be set to an instance of {}".format(type(value)))
        if self.choices is not None:
            if value not in self.choices:
                msg_where = self._err_msg_where(type(instance).__name__)
                raise ValueError(msg_where + "value {} must be one of {}".format(value, self.choices))


    def _assert_have_field_name(self):
        if self.field_name is None:
            # this is the case if the descriptor is not in a class of type MetaType
            raise AttributeError("Declarative attributes can only be declared in subclasses of HasTraits")


    def __get__(self, instance, owner):
        # type: (typing.Any, type) -> typing.Any
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
        self._assert_have_field_name()
        if self.readonly:
            raise AttributeError("can't set readonly attribute")
        self._validate_set(instance, value)

        instance.__dict__[self.field_name] = value


    def __delete__(self, instance):
        msg_where = self._err_msg_where(type(instance).__name__)
        raise AttributeError(msg_where + "can't be deleted")


    def __str__(self):
        return '{}(field_type={}, default={}, required={})'.format(
            type(self).__name__, self.field_type, self.default, self.required)

    def __setattr__(self, key, value):
        if getattr(self, 'owner', None) is not None:
            raise AttributeError("can't change any field after the Attr has been bound to a class")
        super(Attr, self).__setattr__(key, value)


def _auto_docstring(namespace):
    """ generate a docstring for the new class in which the Attrs are documented """
    doc = ['declarative attr on class']

    for k, v in namespace.iteritems():
        if isinstance(v, Attr):
            doc.append(str(v))

    return namespace.get('__doc__', '') + '\n'.join(doc)



class MetaType(abc.ABCMeta):
    """
    Metaclass for the declarative traits.
    We inherit ABCMeta so that the users may use @abstractmethod without having to
    deal with 2 meta-classes.
    Even though we do this we don't support the dynamic registration of subtypes to these abc's
    todo: Review if supporting abstract methods outweighs the complexity of python abc's.
    """
    # This is a python metaclass.
    # For an introduction see https://docs.python.org/2/reference/datamodel.html

    # here to avoid some hasattr; is None etc checks. And to make pycharm happy
    # should be harmless and shadowed by _declarative_attrs on the returned classes
    _own_declarative_attrs = ()  # type: typing.Tuple[str] # name of all declarative fields on this class

    # A record of all the classes we have created, todo: should this hold weakrefs? are classes ever deleted in tvb?
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
        ret = []
        for super_cls in cls.mro():
            if isinstance(super_cls, MetaType):
                for attr_name in getattr(super_cls, '_own_declarative_attrs'):
                    if attr_name not in ret:  # attr was overridden, don't duplicate
                        ret.append(attr_name)
        return tuple(ret)

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

        for k, v in namespace.iteritems():
            if isinstance(v, Attr):
                attrs.append(k)

        # record the names of the declarative attrs in the _own_declarative_attrs field
        if '_own_declarative_attrs' in namespace:
            raise TypeError('class attribute _own_declarative_attrs is reserved in traited classes')

        namespace['_own_declarative_attrs'] = tuple(attrs)
        namespace['__doc__'] = _auto_docstring(namespace)
        # construct the class
        cls = super(MetaType, mcs).__new__(mcs, type_name, bases, namespace)

        # inform the Attr instances about the class their are bound to
        for attr_name in attrs:
            v = namespace[attr_name]
            v.field_name = attr_name
            v.owner = cls
            v._post_bind_validate(type_name)

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
                log.warning('attribute {}.{} = {} is required. '
                            'Initialize it in __init__ or declare a default '
                            .format(cls.__name__, k, attr))


    def __call__(cls, *args, **kwargs):
        """
        Checks that __init__ has initialized required fields
        """
        # call __init__
        instance = super(MetaType, cls).__call__(*args, **kwargs)
        cls._post_init_validate_(instance)
        return instance


    def __setattr__(self, key, value):
        """
        Complain if TraitedClass.a = Attr()
        todo: review this
        """
        if isinstance(value, Attr):
            log.warning('dynamically assigned Attributes are not supported')
        super(MetaType, self).__setattr__(key, value)


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
                raise TypeError('Valid kwargs for type {!r} are: {}. You have given: {!r}'
                                .format(cls, repr(cls.declarative_attrs), k))
            setattr(self, k, v)

    def __str__(self):
        cls = type(self)
        result = ['{} ('.format(self.__class__.__name__)]
        for aname in cls.declarative_attrs:
            attr_field = getattr(self, aname)
            # str would be pretty. but recursive types will stack overflow then
            # use serialization.to_str
            attr_repr = repr(attr_field).splitlines()
            attr_repr = attr_repr[:1] + ['  ' + s for s in attr_repr[1:]]
            attr_repr = '\n'.join(attr_repr)
            result.append('  {} = {},'.format(aname, attr_repr))
        result.append(')')
        return '\n'.join(result)

    def configure(self, *args, **kwargs):
        """
        This is here only because a lot of code relies on configure calls.
        This is the default do nothing base implementation
        todo: deleteme
        """
