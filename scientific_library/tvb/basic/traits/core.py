# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
All problems in computer science can be solved by another layer of
indirection, except for the problem of too many layers of indirection.


Traits overview
---------------

Traits classes are separated into two modules:

core:
    TraitsInfo          meta-data container for traits system
    MetaType            base class for traited class creation
    Type                base traited class 

mapped:
    MappedType             basic class for traited class mapped to db
      * (Type)             traits mapped to columns
      * (MappedType)       traits mapped to column(foreignkey) -> table


Traits metadata
---------------

While the traits of a class can be declared nearly arbitrarily, there are
intrinsic pieces of information in the meta-data required to make the traits
system work:

        doc         string      long description, possibly longer string
        label       string      short name appearing in UI
        default     object      default value of attribute
        required    boolean     determines whether must be set before storage
        range       Range       helps validate or specify parameter variation

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: marmaduke <duke@eml.cc>

"""

import abc
import six
from copy import deepcopy, copy
from tvb.basic.traits.util import get, Args, TypeRegister, ispublic
from tvb.basic.logger.builder import get_logger


LOG = get_logger(__name__)

KWARG_SELECT_MULTIPLE = 'select_multiple'
KWARG_ORDER = 'order'                   # -1 value means hidden from UI
KWARG_AVOID_SUBCLASSES = 'fixed_type'   # When set on a traited attr, no subclasses will be returned
KWARG_FILE_STORAGE = 'file_storage'
KWARG_REQUIRED = 'required'
KWARG_FILTERS_UI = 'filters_ui'
KWARG_FILTERS_BACKEND = 'filters_backend'
KWARG_OPTIONS = 'options'               # Used for Enumerate basic type
KWARG_STORAGE_PATH = 'storage_path'
KWARS_USE_STORAGE = 'use_storage'
KWARS_STORED_METADATA = 'stored_metadata'
KWARGS_LOAD_DEFAULT = "load_default"

FILE_STORAGE_DEFAULT = 'HDF5'
FILE_STORAGE_EXPAND = 'expandable_HDF5'
FILE_STORAGE_NONE = 'None'

SPECIAL_KWDS = ['bind',                 # set and used internally by the traiting mechanism
                'doc', 'label', KWARG_REQUIRED, 'locked', 'default', 'range',
                'configurable_noise', KWARG_OPTIONS, KWARS_STORED_METADATA,
                KWARG_AVOID_SUBCLASSES, KWARG_FILTERS_UI, KWARG_FILTERS_BACKEND, KWARG_SELECT_MULTIPLE, KWARG_ORDER,
                KWARG_FILE_STORAGE, KWARS_USE_STORAGE, KWARGS_LOAD_DEFAULT]


## Module global used by MetaType.
TYPE_REGISTER = TypeRegister()



class TraitsInfo(dict):
    """
    TraitsInfo is a container for information related to the owner
    class and its traited attributes that is used by the traits
    classes. It is needed because many of the attribute names used
    by traits, e.g. data & bound, may mean other things to other
    classes and we can't step on their toes.

    TraitsInfo is a dictionary of the owner's traited attributes, and its other attributes are:

    - name - name of attribute on owner class
    - bound - whether the trait is bound as data descriptor
    - wraps - class this trait wraps
    - inits - a namedtuple of positional and keyword arguments
              given to intialize the trait instance
    - value - instance value of trait, equal to trait if wraps==None
    - defaults - default args to wraps' constructor

    """


    def __init__(self, trait, name='<no name!>', bound=False, wraps=None,
                 inits=Args((), {}), value=None, wraps_defaults=()):
        self.name = name
        self.bound = bound
        self.wraps = wraps
        self.wraps_defaults = wraps_defaults
        self.inits = inits
        self.value = value


    @property
    def file_storage(self):
        if KWARG_FILE_STORAGE not in self.inits.kwd:
            return FILE_STORAGE_DEFAULT
        return self.inits.kwd[KWARG_FILE_STORAGE]


    @property
    def order_number(self):
        if KWARG_ORDER not in self.inits.kwd:
            return 0
        return self.inits.kwd[KWARG_ORDER]


    @property
    def required(self):
        if KWARG_REQUIRED not in self.inits.kwd:
            return True
        return self.inits.kwd[KWARG_REQUIRED]


    @property
    def use_storage(self):
        if KWARS_USE_STORAGE not in self.inits.kwd:
            return True
        return self.inits.kwd[KWARS_USE_STORAGE]


    @property
    def stored_metadata(self):
        if KWARS_STORED_METADATA not in self.inits.kwd:
            return None
        return self.inits.kwd[KWARS_STORED_METADATA]


    @property
    def range_interval(self):
        if 'range' not in self.inits.kwd:
            return None
        return self.inits.kwd['range']


    @property
    def select_multiple(self):
        if KWARG_SELECT_MULTIPLE in self.inits.kwd:
            return self.inits.kwd[KWARG_SELECT_MULTIPLE]
        return False


    def __repr__(self):
        return 'TraitsInfo(%r)' % (super(TraitsInfo, self).__repr__(), )


    def copy(self):
        """
        Create a copy for current Traits.
        """
        new_value = deepcopy(self.value)
        copyed = TraitsInfo(new_value, self.name, self.bound,
                            self.wraps, self.inits, wraps_defaults=self.wraps_defaults)
        for key, value in six.iteritems(self):
            copyed[key] = copy(value)
        return copyed



class Type(object):
    """ Type is redefined below. This is here so MetaType can refer to Type """



class MetaType(abc.ABCMeta):
    """

    The MetaType class augments the class creation and instantiation of all the
    types in the Traits system. See the doc-strings of the methods for more
    details:

        __new__  - creates a class 
        __call__ - create a class instance

    While the basic Traits mechanisms are described and implemented in this
    class, see DeclarativeMetaType for implementation and description of 
    database mapping of Traits classes.

    """


    def __new__(mcs, name, bases, dikt):
        """
        MetaType.__new__ creates a new class, but is typically involved
        *implicitly* either by declaring the __metaclass__ attribute

            class Type(object):
                __metaclass__ = MetaType

        or by subclassing Type:

            class Foo(Type):
                pass

        but in both cases, this method is called to produce the new class object.

        To setup a trait attribute, we :
            - check if it's a type, if so, instantiate
            - tell the attr it's name on owner class 
            - setup private attr with attr.value 
            - augment owner class doc-string with trait description 
            - add trait to information on class 
        """

        # if we're wrapping a class, pop that out
        wraps = dikt.pop('wraps', set([]))
        wraps_defaults = dikt.pop('defaults', ())

        # make new class
        newcls = super(MetaType, mcs).__new__(mcs, name, bases, dikt)
        # add new class to type register
        TYPE_REGISTER.append(newcls)

        # prep class doc string
        doc = get(dikt, '__doc__', 'traited class ' + name)
        doc += "\n    **traits on this class:**\n"

        # build traits info on newcls' attrs
        if hasattr(newcls, 'trait'):
            trait = newcls.trait.copy()
        else:
            trait = TraitsInfo(newcls)

        for key in list(filter(ispublic, dir(newcls))):
            attr = getattr(newcls, key)
            if isinstance(attr, MetaType) or isinstance(attr, Type):
                if isinstance(attr, MetaType):
                    attr = attr()
                try:
                    attr = deepcopy(attr)
                except Exception:
                    attr = copy(attr)
                attr.trait.name = key
                setattr(newcls, key, attr)
                doc += "\n\t``%s`` (%s)\n" % (key, str(attr.trait.inits.kwd.get('label', "")))
                doc += "\t\t| %s\n" % str(attr.trait.inits.kwd.get('doc', "")).replace("\n", " ")
                doc += "\t\t| ``default``:  %s \n" % str(attr.trait.inits.kwd.get('default', None)).replace("\n", " ")
                specified_range = attr.trait.inits.kwd.get('range', None)
                if specified_range:
                    doc += "\t\t| ``range``: low = %s ; high = %s \n\t\t\n" % (str(specified_range.lo),
                                                                               str(specified_range.hi))
                trait[key] = attr

        # add info to new class
        if wraps:
            trait.wraps = wraps
        if wraps_defaults:
            trait.wraps_defaults = wraps_defaults
        newcls.trait = trait

        # bind traits unless told otherwise
        for attr in newcls.trait.values():
            attr.trait.bound = attr.trait.inits.kwd.get('bind', True)

        newcls.__doc__ = doc
        return newcls


    def __call__(cls, *args, **kwds):
        """
        MetaType.__call__ method wraps ncs.__init__(ncs.__new__(*, **), *, **),
        and is implicitly called when the class __init__()s.

        b.Range(*args, **kwds) ->
        b.Range.__init__(b.Range.__new__(MetaType.__call__(b.Range, *args, **kwds), *, **), *, **)

        When creating instances of Traits classes, we

            - if wrapping, try to instantiation wrapped class 
            - check keyword arguments, use to initialize trait attributes
            - record all other keyword args for later use 
            - create class instance 
            - return instance updated with information
        """

        inits = Args(args, kwds.copy())
        if 'default' in kwds:
            value = kwds.pop('default')
            if isinstance(value, MetaType):
                value = value()

        elif cls.trait.wraps:
            wrapped_callable = cls.trait.wraps[0] if isinstance(cls.trait.wraps, tuple) else cls.trait.wraps
            # no args, and we have defaults
            if cls.trait.wraps_defaults:
                _args, _kwds = cls.trait.wraps_defaults
                value = wrapped_callable(*_args, **_kwds)
            # else default constructor, no args
            else:
                value = wrapped_callable()
        else:
            value = None

        kwdtraits = {}
        for key in set(kwds.keys()) & set(cls.trait.keys()):
            kwdtraits[key] = kwds[key]
            del kwds[key]

        options = kwds.get(KWARG_OPTIONS, None)

        # discard kwds to be passed for instantiation
        [kwds.pop(key, None) for key in SPECIAL_KWDS]

        ## 1. - Call __init__ to get a new instance (includes initialization of SqlAlchemy fields)
        try:
            inst = super(MetaType, cls).__call__(*args, **kwds)
        except Exception:
            if not args and not kwds:
                raise
            else:
                # most likely case is that a keyword argument is misspelled.
                msg = "Couldn't create instance of %s with args: %r, %r."
                msg %= cls, args, kwds
                LOG.exception(msg)
                raise TypeError(msg)

        ## 2. - Populate default fields from Trait arguments:
        inst.trait = cls.trait.copy()
        inst.trait.options = options
        inst.trait.value = deepcopy(value)
        inst.trait.inits = inits

        for name, attr in six.iteritems(inst.trait):
            try:
                setattr(inst, name, deepcopy(attr.trait.value))
            except Exception:
                LOG.exception("Could not set attribute '" + name + "' on " + str(inst.__class__.__name__))
                raise

        ## 3. - Load console defaults, in case requested:
        if KWARGS_LOAD_DEFAULT in inits.kwd and inits.kwd[KWARGS_LOAD_DEFAULT]:
            cls.from_file(instance=inst)

        ## 4. - Overwrite with **kwargs from constructor call:
        for name, attr in six.iteritems(kwdtraits):
            try:
                setattr(inst, name, attr)
            except Exception:
                LOG.exception("Could not set kw-given attribute '" + name + "' on " + str(inst.__class__.__name__))
                raise

        # The owner class, if any, will set this to true, see metatype.__new__
        inst.trait.bound = False

        LOG.debug('%s initialized', inst)

        return inst



class Type(object):
    """
    Type class provides a base class for dataTypes and the attributes on dataTypes.

    When a Type instance is an attribute of a class and self.bound is True, the
    instance will act as a data descriptor, setting/getting its corresponding
    value on the owner class.

    In the case of sql'ed values, names are coordinated such that the private
    value (```obj._name```) of the public attr (```obj.name```) on the owner
    class used by the Type instance is actually the corresponding sqlAlchemy
    data descriptor as generated by the value of 'sql' keyword to the Type
    instance __init__.
    """

    __metaclass__ = MetaType
    _summary_info = None

    def __get__(self, inst, cls):
        """
        When an attribute of Type is retrieved on another class.
        """
        if inst is None:
            return self
        if self.trait.bound:
            if hasattr(inst, '_' + self.trait.name):
                ## Return simple DB field or cached value
                return getattr(inst, '_' + self.trait.name)
            else:
                return None
        else:
            return self


    def __set__(self, inst, value):
        """
        When an attribute of Type is set on another class
        """
        if self.trait.bound:
            ## First validate that the given value is compatible with the current attribute definition
            accepted_types = [type(self), type(None)]
            if isinstance(self.trait.wraps, tuple):
                accepted_types.extend(self.trait.wraps)
            else:
                accepted_types.append(self.trait.wraps)

            if (type(value) in accepted_types or isinstance(value, type(self))
                or (isinstance(value, (list, tuple)) and self.trait.select_multiple)):
                self._put_value_on_instance(inst, value)
            else:
                msg = 'expected type %s, received type %s' % (type(self), type(value))
                LOG.error(msg)
                raise AttributeError(msg)


    def _put_value_on_instance(self, inst, value):
        """
        Is the ultimate method called by __set__ implementations.
        We write it separately here, because subclasses might need to call this separately, 
        without the __set__ default value validation.
        """
        setattr(inst, '_' + self.trait.name, value)
        inst.trait[self.trait.name].value = value


    def __repr__(self):
        """
        Type.repr builds a useful representation of itself, which can be 
        configured with values in config:
        """
        trait = self.trait
        rep = self.__class__.__name__ + "("
        objstr = object.__repr__(self)
        value = objstr if self is trait.value else trait.value

        reprinfo = {'value': repr(value),
                    'bound': repr(trait.bound)}
        if trait.wraps:
            reprinfo['wraps'] = repr(trait.wraps)
        if trait.bound:
            reprinfo['name'] = repr(trait.name)

        reprstr = ['%s=%s' % (k, v) for k, v in reprinfo.items()]
        return rep + ', '.join(reprstr) + ')'


    def configure(self):
        """
        Call this method to process linked attributes on DataType.
        This will be called before storing entity in DB.
        """
        pass


    @property
    def summary_info(self):
        """
        For a particular DataType, return a dictionary of label: value, 
        to describe the entity from scientific point of view.
        """
        if self._summary_info is None and hasattr(self, "_find_summary_info"):
            self._summary_info = self._find_summary_info()
        return self._summary_info

    # Used by IPython Notebook
    def _repr_html_(self):
        "Generate HTML repr for use in IPython notebook."
        info = self.summary_info
        if info is None or len(info) == 0:
            info = {key: getattr(self, key) for key in self.trait.keys()}
        html = ['<table width=100%>']
        row_fmt = '<tr><td>%s</td><td>%s</td></tr>'
        for key, value in info.items():
            html.append(row_fmt % (key, value))
        return ''.join(html)

    def _find_summary_info(self):
        """
        To be implemented in every subclass.
        """
        return None


TypeBase = Type
