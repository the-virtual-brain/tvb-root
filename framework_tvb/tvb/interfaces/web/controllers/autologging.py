# -*- coding: utf-8 -*-

# Copyright 2013, 2015, 2016, 2018, 2019 Matthew Zipay.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# http://ninthtest.info/python-autologging/

"""
Edited by Bogdan Valean <bogdan.valean@codemart.ro> : 29.05.2020
- Use internal TVB Logger by default.
- Add logged user GID to the logging message
"""

__author__ = "Matthew Zipay (mattzATninthtestDOTinfo)"
__version__ = "1.3.2"

import logging
import platform
import sys
import warnings
from functools import wraps
from inspect import isclass, isgenerator, isroutine
from types import FunctionType

from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.interfaces.web.controllers.common import get_logged_user

#: A custom tracing log level, lower in severity than :attr:`logging.DEBUG`.
TRACE = 1
logging.addLevelName(TRACE, "TRACE")

LOGGER = get_logger("tvb_user_actions")
LOGGER.setLevel(TRACE)

# BEGIN Jython/IronPython detection
# (this needs to be implemented consistently w/r/t Aglyph's aglyph._compat)
try:
    _py_impl = platform.python_implementation()
except:
    _py_impl = "Python"

_is_jython = \
    _py_impl == "Jython" and getattr(sys, "JYTHON_JAR", None) is not None

try:
    import clr

    clr.AddReference("System")
    _has_clr = True
except:
    _has_clr = False

_is_ironpython = _py_impl == "IronPython" and _has_clr
# END Jython/IronPython detection

__all__ = [
    "TRACE",
    "logged",
    "traced",
    "install_traced_noop",
]


def logged(obj):
    """Add a logger member to a decorated class or function.

    :arg obj:
       the class or function object being decorated, or an optional
       :class:`logging.Logger` object to be used as the parent logger
       (instead of the default module-named logger)
    :return:
       *obj* if *obj* is a class or function; otherwise, if *obj* is a
       logger, return a lambda decorator that will in turn set the
       logger attribute and return *obj*

    If *obj* is a :obj:`class`, then ``obj.__log`` will have the logger
    name "<module-name>.<class-name>":

    >>> import sys
    >>> logging.basicConfig(
    ...     level=logging.DEBUG, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @logged
    ... class Sample:
    ...
    ...     def test(self):
    ...         self.__log.debug("This is a test.")
    ...
    >>> Sample().test()
    DEBUG:autologging.Sample:test:This is a test.

    .. note::
       Autologging will prefer to use the class's ``__qualname__`` when
       it is available (Python 3.3+). Otherwise, the class's
       ``__name__`` is used. For example::

          class Outer:

             @logged
             class Nested: pass

       Under Python 3.3+, ``Nested.__log`` will have the name
       "autologging.Outer.Nested", while under Python 2.7 or 3.2, the
       logger name will be "autologging.Nested".

    .. versionchanged:: 0.4.0
       Functions decorated with ``@logged`` use a *single* underscore
       in the logger variable name (e.g. ``my_function._log``) rather
       than a double underscore.

    If *obj* is a function, then ``obj._log`` will have the logger name
    "<module-name>":

    >>> import sys
    >>> logging.basicConfig(
    ...     level=logging.DEBUG, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @logged
    ... def test():
    ...     test._log.debug("This is a test.")
    ...
    >>> test()
    DEBUG:autologging:test:This is a test.

    .. note::
       Within a logged function, the ``_log`` attribute must be
       qualified by the function name.

    If *obj* is a :class:`logging.Logger` object, then that logger is
    used as the parent logger (instead of the default module-named
    logger):

    >>> import sys
    >>> logging.basicConfig(
    ...     level=logging.DEBUG, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @logged(logging.getLogger("test.parent"))
    ... class Sample:
    ...     def test(self):
    ...         self.__log.debug("This is a test.")
    ...
    >>> Sample().test()
    DEBUG:test.parent.Sample:test:This is a test.

    Again, functions are similar:

    >>> import sys
    >>> logging.basicConfig(
    ...     level=logging.DEBUG, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @logged(logging.getLogger("test.parent"))
    ... def test():
    ...     test._log.debug("This is a test.")
    ...
    >>> test()
    DEBUG:test.parent:test:This is a test.

    .. note::
       For classes, the logger member is made "private" (i.e. ``__log``
       with double underscore) to ensure that log messages that include
       the *%(name)s* format placeholder are written with the correct
       name.

       Consider a subclass of a ``@logged``-decorated parent class. If
       the subclass were **not** decorated with ``@logged`` and could
       access the parent's logger member directly to make logging
       calls, those log messages would display the name of the
       parent class, not the subclass.

       Therefore, subclasses of a ``@logged``-decorated parent class
       that wish to use a provided ``self.__log`` object must themselves
       be decorated with ``@logged``.

    .. warning::
       Although the ``@logged`` and ``@traced`` decorators will "do the
       right thing" regardless of the order in which they are applied to
       the same function, it is recommended that ``@logged`` always be
       used as the innermost decorator::

          @traced
          @logged
          def my_function():
              my_function._log.info("message")

       This is because ``@logged`` simply sets the ``_log`` attribute
       and then returns the original function, making it "safe" to use
       in combination with any other decorator.

    .. note::
       Both `Jython <http://www.jython.org/>`_ and
       `IronPython <http://ironpython.net/>`_ report an "internal" class
       name using its mangled form, which will be reflected in the
       default logger name.

       For example, in the sample code below, both Jython and IronPython
       will use the default logger name "autologging._Outer__Nested"
       (whereas CPython/PyPy/Stackless would use "autologging.__Nested"
       under Python 2 or "autologging.Outer.__Nested" under Python 3.3+)
       ::

          class Outer:
             @logged
             class __Nested:
                pass

    .. warning::
       `IronPython <http://ironpython.net/>`_ does not fully support
       frames (even with the -X:FullFrames option), so you are likely to
       see things like misreported line numbers and "<unknown file>" in
       log records emitted when running under IronPython.

    """
    if isinstance(obj, logging.Logger):  # `@logged(logger)'
        return lambda class_or_fn: _add_logger_to(
            class_or_fn,
            logger_name=_generate_logger_name(
                class_or_fn, parent_name=obj.name))
    else:  # `@logged'
        return _add_logger_to(obj)


def traced(*args, **keywords):
    """Add call and return tracing to an unbound function or to the
    methods of a class.

    The arguments to ``traced`` differ depending on whether it is being
    used to trace an unbound function or the methods of a class:

    .. rubric:: Trace an unbound function using the default logger

    :arg func: the unbound function to be traced

    By default, a logger named for the function's module is used:

    >>> import sys
    >>> logging.basicConfig(
    ...     level=TRACE, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @traced
    ... def func(x, y):
    ...     return x + y
    ...
    >>> func(7, 9)
    TRACE:autologging:func:CALL *(7, 9) **{}
    TRACE:autologging:func:RETURN 16
    16

    .. rubric:: Trace an unbound function using a named logger

    :arg logging.Logger logger:
       the parent logger used to trace the unbound function

    >>> import sys
    >>> logging.basicConfig(
    ...     level=TRACE, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @traced(logging.getLogger("my.channel"))
    ... def func(x, y):
    ...     return x + y
    ...
    >>> func(7, 9)
    TRACE:my.channel:func:CALL *(7, 9) **{}
    TRACE:my.channel:func:RETURN 16
    16

    .. rubric:: Trace default methods using the default logger

    :arg class_: the class whose methods will be traced

    By default, all "public", "_nonpublic", and "__internal" methods, as
    well as the special "__init__" and "__call__" methods, will be
    traced. Tracing log entries will be written to a logger named for
    the module and class:

    >>> import sys
    >>> logging.basicConfig(
    ...     level=TRACE, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @traced
    ... class Class:
    ...     def __init__(self, x):
    ...         self._x = x
    ...     def public(self, y):
    ...         return self._x + y
    ...     def _nonpublic(self, y):
    ...         return self._x - y
    ...     def __internal(self, y=2):
    ...         return self._x ** y
    ...     def __repr__(self):
    ...         return "Class(%r)" % self._x
    ...     def __call__(self):
    ...         return self._x
    ...
    >>> obj = Class(7)
    TRACE:autologging.Class:__init__:CALL *(7,) **{}
    >>> obj.public(9)
    TRACE:autologging.Class:public:CALL *(9,) **{}
    TRACE:autologging.Class:public:RETURN 16
    16
    >>> obj._nonpublic(5)
    TRACE:autologging.Class:_nonpublic:CALL *(5,) **{}
    TRACE:autologging.Class:_nonpublic:RETURN 2
    2
    >>> obj._Class__internal(y=3)
    TRACE:autologging.Class:__internal:CALL *() **{'y': 3}
    TRACE:autologging.Class:__internal:RETURN 343
    343
    >>> repr(obj) # not traced by default
    'Class(7)'
    >>> obj()
    TRACE:autologging.Class:__call__:CALL *() **{}
    TRACE:autologging.Class:__call__:RETURN 7
    7

    .. note::
       When the runtime Python version is >= 3.3, the *qualified* class
       name will be used to name the tracing logger (i.e. a nested class
       will write tracing log entries to a logger named
       "module.Parent.Nested").

    .. rubric:: Trace default methods using a named logger

    :arg logging.Logger logger:
       the parent logger used to trace the methods of the class

    By default, all "public", "_nonpublic", and "__internal" methods, as
    well as the special "__init__" method, will be traced. Tracing log
    entries will be written to the specified logger:

    >>> import sys
    >>> logging.basicConfig(
    ...     level=TRACE, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @traced(logging.getLogger("my.channel"))
    ... class Class:
    ...     def __init__(self, x):
    ...         self._x = x
    ...     def public(self, y):
    ...         return self._x + y
    ...     def _nonpublic(self, y):
    ...         return self._x - y
    ...     def __internal(self, y=2):
    ...         return self._x ** y
    ...     def __repr__(self):
    ...         return "Class(%r)" % self._x
    ...     def __call__(self):
    ...         return self._x
    ...
    >>> obj = Class(7)
    TRACE:my.channel.Class:__init__:CALL *(7,) **{}
    >>> obj.public(9)
    TRACE:my.channel.Class:public:CALL *(9,) **{}
    TRACE:my.channel.Class:public:RETURN 16
    16
    >>> obj._nonpublic(5)
    TRACE:my.channel.Class:_nonpublic:CALL *(5,) **{}
    TRACE:my.channel.Class:_nonpublic:RETURN 2
    2
    >>> obj._Class__internal(y=3)
    TRACE:my.channel.Class:__internal:CALL *() **{'y': 3}
    TRACE:my.channel.Class:__internal:RETURN 343
    343
    >>> repr(obj) # not traced by default
    'Class(7)'
    >>> obj()
    TRACE:my.channel.Class:__call__:CALL *() **{}
    TRACE:my.channel.Class:__call__:RETURN 7
    7

    .. rubric:: Trace specified methods using the default logger

    :arg tuple method_names:
       the names of the methods that will be traced

    Tracing log entries will be written to a logger named for the
    module and class:

    >>> import sys
    >>> logging.basicConfig(
    ...     level=TRACE, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @traced("public", "__internal")
    ... class Class:
    ...     def __init__(self, x):
    ...         self._x = x
    ...     def public(self, y):
    ...         return self._x + y
    ...     def _nonpublic(self, y):
    ...         return self._x - y
    ...     def __internal(self, y=2):
    ...         return self._x ** y
    ...     def __repr__(self):
    ...         return "Class(%r)" % self._x
    ...     def __call__(self):
    ...         return self._x
    ...
    >>> obj = Class(7)
    >>> obj.public(9)
    TRACE:autologging.Class:public:CALL *(9,) **{}
    TRACE:autologging.Class:public:RETURN 16
    16
    >>> obj._nonpublic(5)
    2
    >>> obj._Class__internal(y=3)
    TRACE:autologging.Class:__internal:CALL *() **{'y': 3}
    TRACE:autologging.Class:__internal:RETURN 343
    343
    >>> repr(obj)
    'Class(7)'
    >>> obj()
    7

    .. warning::
       When method names are specified explicitly via *args*,
       Autologging ensures that each method is actually defined in
       the body of the class being traced. (This means that inherited
       methods that are not overridden are **never** traced, even if
       they are named explicitly in *args*.)

       If a defintion for any named method is not found in the class
       body, either because the method is inherited or because the
       name is misspelled, Autologging will issue a :exc:`UserWarning`.

       If you wish to trace a method from a super class, you have two
       options:

       1. Use ``traced`` to decorate the super class.
       2. Override the method and trace it in the subclass.

    .. note::
       When the runtime Python version is >= 3.3, the *qualified* class
       name will be used to name the tracing logger (i.e. a nested class
       will write tracing log entries to a logger named
       "module.Parent.Nested").

    .. rubric:: Trace specified methods using a named logger

    :arg logging.Logger logger:
       the parent logger used to trace the methods of the class
    :arg tuple method_names:
       the names of the methods that will be traced

    >>> import sys
    >>> logging.basicConfig(
    ...     level=TRACE, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @traced(logging.getLogger("my.channel"), "public", "__internal")
    ... class Class:
    ...     def __init__(self, x):
    ...         self._x = x
    ...     def public(self, y):
    ...         return self._x + y
    ...     def _nonpublic(self, y):
    ...         return self._x - y
    ...     def __internal(self, y=2):
    ...         return self._x ** y
    ...     def __repr__(self):
    ...         return "Class(%r)" % self._x
    ...     def __call__(self):
    ...         return self._x
    ...
    >>> obj = Class(7)
    >>> obj.public(9)
    TRACE:my.channel.Class:public:CALL *(9,) **{}
    TRACE:my.channel.Class:public:RETURN 16
    16
    >>> obj._nonpublic(5)
    2
    >>> obj._Class__internal(y=3)
    TRACE:my.channel.Class:__internal:CALL *() **{'y': 3}
    TRACE:my.channel.Class:__internal:RETURN 343
    343
    >>> repr(obj) # not traced by default
    'Class(7)'
    >>> obj()
    7

    .. warning::
       When method names are specified explicitly via *args*,
       Autologging ensures that each method is actually defined in
       the body of the class being traced. (This means that inherited
       methods that are not overridden are **never** traced, even if
       they are named explicitly in *args*.)

       If a defintion for any named method is not found in the class
       body, either because the method is inherited or because the
       name is misspelled, Autologging will issue a :exc:`UserWarning`.

       If you wish to trace a method from a super class, you have two
       options:

       1. Use ``traced`` to decorate the super class.
       2. Override the method and trace it in the subclass.

    .. rubric:: Exclude specified methods from tracing

    .. versionadded:: 1.3.0

    :arg tuple method_names:
       the names of the methods that will be excluded from tracing
    :keyword bool exclude:
       ``True`` to cause the method names list to be interpreted as
       an exclusion list (``False`` is the default, and causes the named
       methods to be **included** as described above)

    The example below demonstrates exclusions using the default logger.

    >>> import sys
    >>> logging.basicConfig(
    ...     level=TRACE, stream=sys.stdout,
    ...     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    >>> @traced("_nonpublic", "__internal", exclude=True)
    ... class Class:
    ...     def __init__(self, x):
    ...         self._x = x
    ...     def public(self, y):
    ...         return self._x + y
    ...     def _nonpublic(self, y):
    ...         return self._x - y
    ...     def __internal(self, y=2):
    ...         return self._x ** y
    ...     def __repr__(self):
    ...         return "Class(%r)" % self._x
    ...     def __call__(self):
    ...         return self._x
    ...
    >>> obj = Class(7)
    >>> obj.public(9)
    TRACE:autologging.Class:public:CALL *(9,) **{}
    TRACE:autologging.Class:public:RETURN 16
    16
    >>> obj._nonpublic(5)
    2
    >>> obj._Class__internal(y=3)
    343
    >>> repr(obj)
    'Class(7)'
    >>> obj()
    TRACE:autologging.Class:__call__:CALL *() **{}
    TRACE:autologging.Class:__call__:RETURN 7
    7

    When method names are excluded via *args* and the *exclude* keyword,
    Autologging **ignores** methods that are not actually defined in the
    body of the class being traced.

    .. warning::
       If an exclusion list causes the list of traceable methods to
       resolve empty, then Autologging will issue a :exc:`UserWarning`.

    .. note::
       When the runtime Python version is >= 3.3, the *qualified* class
       name will be used to name the tracing logger (i.e. a nested class
       will write tracing log entries to a logger named
       "module.Parent.Nested").

    .. note::
       When tracing a class, if the default (class-named) logger is
       used **and** the runtime Python version is >= 3.3, then the
       *qualified* class name will be used to name the tracing logger
       (i.e. a nested class will write tracing log entries to a logger
       named "module.Parent.Nested").

    .. note::
       If method names are specified when decorating a function, a
       :exc:`UserWarning` is issued, but the methods names are ignored
       and the function is traced as though the method names had not
       been specified.

    .. note::
       Both `Jython <http://www.jython.org/>`_ and
       `IronPython <http://ironpython.net/>`_ report an "internal" class
       name using its mangled form, which will be reflected in the
       default tracing logger name.

       For example, in the sample code below, both Jython and IronPython
       will use the default tracing logger name
       "autologging._Outer__Nested" (whereas CPython/PyPy/Stackless
       would use "autologging.__Nested" under Python 2 or
       "autologging.Outer.__Nested" under Python 3.3+)::

          class Outer:
             @traced
             class __Nested:
                pass

    .. warning::
       Neither `Jython <http://www.jython.org/>`_ nor
       `IronPython <http://ironpython.net/>`_ currently implement the
       ``function.__code__.co_lnotab`` attribute, so the last line
       number of a function cannot be determined by Autologging.

    .. versionchanged:: 1.3.1
       Due to unavoidable inconsistencies in line number tracking across
       Python variants (see
       `issues/6 <https://github.com/mzipay/Autologging/issues/6>`_, as
       of version 1.3.1 and until further notice Autologging will only
       record the first line number of the function being traced in all
       tracing CALL and RETURN records.
       (Note that YIELD tracing records for generator iterators will
       continue to record the correct line number on variants other than
       IronPython.)

    """
    obj = args[0] if args else None
    if obj is None:
        # treat `@traced()' as equivalent to `@traced'
        return traced

    if isclass(obj):  # `@traced' class
        return _install_traceable_methods(obj,
                                          exclude=keywords.get("exclude", False))
    elif isroutine(obj):  # `@traced' function
        LOGGER.setLevel(TRACE)
        return _make_traceable_function(
            obj, LOGGER)
    elif isinstance(obj, logging.Logger):
        # may be decorating a class OR a function
        method_names = args[1:]
        exclude = keywords.get("exclude", False)

        def traced_decorator(class_or_fn):
            if isclass(class_or_fn):
                # `@traced(logger)' or `@traced(logger, "method", ..)' class
                return _install_traceable_methods(
                    class_or_fn, *method_names, exclude=exclude,
                    logger=logging.getLogger(
                        _generate_logger_name(
                            class_or_fn, parent_name=obj.name)))
            else:  # `@traced(logger)' function
                if method_names:
                    warnings.warn(
                        "ignoring method names for @traced function %s.%s" %
                        (class_or_fn.__module__, class_or_fn.__name__))
                return _make_traceable_function(class_or_fn, obj)

        return traced_decorator
    else:  # `@traced("method_name1", ..)' class
        method_names = args[:]
        exclude = keywords.get("exclude", False)
        return lambda class_: _install_traceable_methods(class_, *method_names,
                                                         exclude=exclude)


__traced_original = traced


def _traced_noop(*args, **keywords):
    """Turn the ``@traced`` decorator into a no-op."""
    obj = args[0] if args else None
    if obj is None:
        # treat `@traced()' as equivalent to `@traced'
        return _traced_noop

    if isclass(obj) or isroutine(obj):  # `@traced' class or function
        return obj
    else:  # `@traced(logger)' or `@traced("method_name1", ..)'
        def traced_noop_decorator(class_or_fn):
            return class_or_fn

        return traced_noop_decorator


def install_traced_noop():
    """Replace the :func:`traced` decorator with an identity (no-op)
    decorator.

    Although the overhead of a ``@traced`` function or method is minimal
    when the :data:`TRACED` log level is disabled, there is still *some*
    overhead (the logging level check, an extra function call).

    If you would like to completely *eliminate* this overhead, call this
    function **before** any classes or functions in your application are
    decorated with ``@traced``. The :func:`traced` decorator will be
    replaced with a no-op decorator that simply returns the class or
    function unmodified.

    .. note::
       The **recommended** way to install the no-op ``@traced``
       decorator is to set the ``AUTOLOGGING_TRACED_NOOP``
       environment variable to any non-empty value.

       If the ``AUTOLOGGING_TRACED_NOOP`` environment variable is
       set to a non-empty value when Autologging is loaded, the
       ``@traced`` no-op will be installed automatically.

    As an alternative to setting the ``AUTOLOGGING_TRACED_NOOP``
    environment variable, you can also call this function directly in
    your application's bootstrap module. For example::

       import autologging

       if running_in_production:
           autologging.install_traced_noop()

    .. warning::
       This function **does not** "revert" any already-``@traced`` class
       or function! It simply replaces the ``autologging.traced`` module
       reference with a no-op.

       For this reason it is imperative that
       ``autologging.install_traced_noop()`` be called **before** the
       ``@traced`` decorator has been applied to any class or function
       in the application. (This is why the ``AUTOLOGGING_TRACED_NOOP``
       environment variable is the recommended approach for installing
       the no-op - it allows Autologging itself to guarantee that the
       no-op is installed before any classes or functions are
       decorated.)

    """
    global traced
    traced = _traced_noop
    logging.getLogger().info("autologging.traced no-op is installed")


if not TvbProfile.current.TRACE_USER_ACTIONS:
    install_traced_noop()


def _generate_logger_name(obj, parent_name=None):
    """Generate the logger name (channel) for a class or function.

    :arg obj: a class or function
    :keyword str parent_name: the name of *obj*'s parent logger
    :rtype: str

    If *parent_name* is not specified, the default is to use *obj*'s
    module name.

    """
    parent_logger_name = parent_name if parent_name else obj.__module__
    return "%s.%s" % (
        parent_logger_name, getattr(obj, "__qualname__", obj.__name__)) \
        if isclass(obj) else parent_logger_name


def _add_logger_to(obj, logger_name=None):
    """Set a :class:`logging.Logger` member on *obj*.

    :arg obj: a class or function object
    :keyword str logger_name: the name (channel) of the logger for *obj*
    :return: *obj*

    If *obj* is a class, the member will be named "__log". If *obj* is a
    function, the member will be named "_log".

    """
    logger = logging.getLogger(
        logger_name if logger_name else _generate_logger_name(obj))

    if isclass(obj):
        setattr(obj, _mangle_name("__log", obj.__name__), logger)
    else:  # function
        obj._log = logger

    return obj


def _make_traceable_function(function, logger):
    """Create a function that delegates to either a tracing proxy or
    the original *function*.

    :arg function:
       an unbound, module-level (or nested) function
    :arg logging.Logger logger: the tracing logger
    :return:
       a function that wraps *function* to provide the call and return
       tracing support

    If *logger* is not enabled for the :attr:`autologging.TRACE`
    level **at the time the returned delegator function is invoked**,
    then the original *function* is called instead of the tracing proxy.

    The overhead that a ``@traced`` function incurs when tracing is
    **disabled** is:

    * the delegator function call itself
    * the ``TRACE`` level check.

    The original *function* is available from the delegator function's
    ``__wrapped__`` attribute.

    """
    proxy = _FunctionTracingProxy(function, logger)

    @wraps(function)
    def autologging_traced_function_delegator(*args, **keywords):
        if logger.isEnabledFor(TRACE):
            # don't access the proxy from closure (IronPython does not manage
            # co_freevars/__closure__ correctly for local vars)
            proxy = autologging_traced_function_delegator._tracing_proxy
            return proxy(function, args, keywords)
        else:
            return function(*args, **keywords)

    autologging_traced_function_delegator._tracing_proxy = proxy

    if not hasattr(autologging_traced_function_delegator, "__wrapped__"):
        # __wrapped__ is only set by functools.wraps() in Python 3.2+
        autologging_traced_function_delegator.__wrapped__ = function

    autologging_traced_function_delegator.__autologging_traced__ = True

    return autologging_traced_function_delegator


# can't use option=<default> keywords with *args in Python 2.7 (see PEP-3102)
def _install_traceable_methods(class_, *method_names, **keywords):
    """Substitute tracing proxy methods for the methods named in
    *method_names* in *class_*'s ``__dict__``.

    :arg class_:
       a class being traced
    :arg tuple method_names:
       the names of the methods to be traced
    :keyword logging.Logger logger:
       the logger to use for tracing
    :keyword bool exclude:
       ``True`` to interpret *method_names* as an **exclusion** list
       rather than an inclusion list

    If *method_names* is empty and the *exclude* keyword is ``False``
    (the default), then all "public", "_nonpublic", and "__internal"
    methods, as well as the special "__init__" and "__call__" methods,
    will be traced by default.

    If the *exclude* keyword is ``True``, then the methods that will be
    traced are the default methods (as identified above) **MINUS** any
    methods named in *method_names*.

    If *logger* is unspecified, a default logger will be used to log
    tracing messages.

    """

    if method_names:
        traceable_method_names = _get_traceable_method_names(method_names,
                                                             class_, exclude=keywords.get("exclude", False))
    else:
        traceable_method_names = _get_default_traceable_method_names(class_)

    # replace each named method with a tracing proxy method
    for method_name in traceable_method_names:
        descriptor = class_.__dict__[method_name]
        descriptor_type = type(descriptor)

        if descriptor_type is FunctionType:
            make_traceable_method = _make_traceable_instancemethod
        elif descriptor_type is classmethod:
            make_traceable_method = _make_traceable_classmethod
        elif descriptor_type is staticmethod:
            make_traceable_method = _make_traceable_staticmethod
        else:
            # should be unreachable, but issue a warning just in case
            warnings.warn("tracing not supported for %r" % descriptor_type)
            continue

        LOGGER.setLevel(TRACE)
        tracing_proxy_descriptor = make_traceable_method(descriptor, LOGGER)

        # class_.__dict__ is a mappingproxy; direct assignment not supported
        setattr(class_, method_name, tracing_proxy_descriptor)

    return class_


def _get_traceable_method_names(method_names, class_, **keywords):
    """Filter (and possibly mangle) *method_names* so that only method
    names actually defined as methods in *cls_dict* remain.

    :arg method_names:
       a sequence of names that should identify methods defined in
       *class_*
    :arg class_: the class being traced
    :keyword bool exclude:
       ``True`` to interpret *method_names* as an **exclusion** list
       rather than an inclusion list
    :return:
       a sequence of names identifying methods that are defined in
       *class_* that will be traced
    :rtype: list

    .. warning::
       A :exc:`UserWarning` is issued if any **included** method named
       in *method_names* is not actually defined in *class_*; or if the
       result of filtering **excluded** methods results in an empty
       list.

    """
    exclude = keywords.get("exclude", False)
    traceable_method_names = []

    if not keywords.get("exclude", False):
        for name in method_names:
            mname = (
                name if not _is_internal_name(name) else
                _mangle_name(name, class_.__name__))

            if isroutine(class_.__dict__.get(mname)):
                traceable_method_names.append(mname)
            else:
                warnings.warn(
                    "%r does not identify a method defined in %s" %
                    (name, class_.__name__))
    else:
        traceable_method_names = [
            name
            for name in _get_default_traceable_method_names(class_)
            if _unmangle_name(name, class_.__name__) not in method_names]
        if not traceable_method_names:
            warnings.warn((
                              "exclude=True with the supplied method names results in NO "
                              "traceable methods for %s") % class_.__name__)

    return traceable_method_names


def _get_default_traceable_method_names(class_):
    """Return all names in *cls_dict* that identify methods.

    :arg class_: the class being traced
    :return:
       a sequence of names identifying methods of *class_* that will be
       traced
    :rtype: list

    """
    default_traceable_method_names = []

    for (name, member) in class_.__dict__.items():
        if isroutine(member) and (
                not _is_special_name(name) or
                name in ("__init__", "__call__")):
            default_traceable_method_names.append(name)

    return default_traceable_method_names


def _is_internal_name(name):
    """Determine whether or not *name* is an "__internal" name.

    :arg str name: a name defined in a class ``__dict__``
    :return: ``True`` if *name* is an "__internal" name, else ``False``
    :rtype: bool

    """
    return name.startswith("__") and not name.endswith("__")


def _mangle_name(internal_name, class_name):
    """Transform *internal_name* (which is assumed to be an "__internal"
    name) into a "_ClassName__internal" name.

    :arg str internal_name:
       the assumed-to-be-"__internal" member name
    :arg str class_name:
       name of the class where *internal_name* is defined
    :return:
       the transformed "_ClassName__internal" name
    :rtype:
       str

    """
    return "_%s%s" % (class_name.lstrip('_'), internal_name)


def _unmangle_name(mangled_name, class_name):
    """Transform *mangled_name* (which is assumed to be a
    "_ClassName__internal" name) into an "__internal" name.

    :arg str mangled_name:
       a mangled "_ClassName__internal" member name
    :arg str class_name:
       name of the class where the (unmangled) name is defined
    :return:
       the transformed "__internal" name
    :rtype:
       str

    """
    return mangled_name.replace("_%s" % class_name.lstrip('_'), "")


def _is_special_name(name):
    """Determine whether or not *name* is a "__special__" name.

    :arg str name: a name defined in a class ``__dict__``
    :return: ``True`` if *name* is a "__special__" name, else ``False``
    :rtype: bool

    """
    return name.startswith("__") and name.endswith("__")


def _make_traceable_instancemethod(unbound_function, logger):
    """Create an unbound function that delegates to either a tracing
    proxy or the original *unbound_function*.

    :arg unbound_function:
       the unbound function for the instance method being traced
    :arg logging.Logger logger: the tracing logger
    :return:
       an unbound function that wraps *unbound_function* to provide the
       call and return tracing support

    If *logger* is not enabled for the :attr:`autologging.TRACE`
    level **at the time the returned delegator function is invoked**,
    then the method for the original *unbound_function* is called
    instead of the tracing proxy.

    The overhead that a ``@traced`` instance method incurs when tracing
    is **disabled** is:

    * the delegator function call itself
    * binding the original *unbound_function* to the instance
    * the ``TRACE`` level check

    The original *unbound_function* is available from the delegator
    function's ``__wrapped__`` attribute.

    """
    # functions have a __get__ method; they can act as descriptors
    proxy = _FunctionTracingProxy(unbound_function, logger)

    @wraps(unbound_function)
    def autologging_traced_instancemethod_delegator(self_, *args, **keywords):
        method = unbound_function.__get__(self_, self_.__class__)
        if logger.isEnabledFor(TRACE):
            # don't access the proxy from closure (IronPython does not manage
            # co_freevars/__closure__ correctly for local vars)
            proxy = \
                autologging_traced_instancemethod_delegator._tracing_proxy
            return proxy(method, args, keywords)
        else:
            return method(*args, **keywords)

    autologging_traced_instancemethod_delegator._tracing_proxy = proxy

    if not hasattr(
            autologging_traced_instancemethod_delegator, "__wrapped__"):
        # __wrapped__ is only set by functools.wraps() in Python 3.2+
        autologging_traced_instancemethod_delegator.__wrapped__ = \
            unbound_function

    autologging_traced_instancemethod_delegator.__autologging_traced__ = True

    return autologging_traced_instancemethod_delegator


def _make_traceable_classmethod(method_descriptor, logger):
    """Create a method descriptor that delegates to either a tracing
    proxy or the original *method_descriptor*.

    :arg method_descriptor:
       the method descriptor for the class method being traced
    :arg logging.Logger logger: the tracing logger
    :return:
       a method descriptor that wraps the *method_descriptor* function
       to provide the call and return tracing support

    If *logger* is not enabled for the :attr:`autologging.TRACE`
    level **at the time the returned delegator method descriptor is
    invoked**, then the method for the original *method_descriptor* is
    called instead of the tracing proxy.

    The overhead that a ``@traced`` class method incurs when tracing is
    **disabled** is:

    * the delegator function call itself
    * binding the original *method_descriptor* to the class
    * the ``TRACE`` level check

    The original *method_descriptor* function is available from the
    delegator method descriptor's ``__func__.__wrapped__`` attribute.

    """
    function = method_descriptor.__func__
    proxy = _FunctionTracingProxy(function, logger)

    @wraps(function)
    def autologging_traced_classmethod_delegator(cls, *args, **keywords):
        method = method_descriptor.__get__(None, cls)
        if logger.isEnabledFor(TRACE):
            # don't access the proxy from closure (IronPython does not manage
            # co_freevars/__closure__ correctly for local vars)
            proxy = autologging_traced_classmethod_delegator._tracing_proxy
            return proxy(method, args, keywords)
        else:
            return method(*args, **keywords)

    autologging_traced_classmethod_delegator._tracing_proxy = proxy

    if not hasattr(autologging_traced_classmethod_delegator, "__wrapped__"):
        # __wrapped__ is only set by functools.wraps() in Python 3.2+
        autologging_traced_classmethod_delegator.__wrapped__ = function

    autologging_traced_classmethod_delegator.__autologging_traced__ = True

    return classmethod(autologging_traced_classmethod_delegator)


def _make_traceable_staticmethod(method_descriptor, logger):
    """Create a method descriptor that delegates to either a tracing
    proxy or the original *method_descriptor*.

    :arg method_descriptor:
       the method descriptor for the static method being traced
    :arg logging.Logger logger: the tracing logger
    :return:
       a method descriptor that wraps the *method_descriptor* function
       to provide the call and return tracing support

    If *logger* is not enabled for the :attr:`autologging.TRACE`
    level **at the time the returned delegator method descriptor is
    invoked**, then the method for the original *method_descriptor* is
    called instead of the tracing proxy.

    The overhead that a ``@traced`` static method incurs when tracing is
    **disabled** is:

    * the delegator function call itself
    * the ``TRACE`` level check

    The original *method_descriptor* function is available from the
    delegator method descriptor's ``__func__.__wrapped__`` attribute.

    """
    autologging_traced_staticmethod_delegator = _make_traceable_function(
        method_descriptor.__func__, logger)

    return staticmethod(autologging_traced_staticmethod_delegator)


class _FunctionTracingProxy(object):
    """Proxy a function invocation to capture and log the call arguments
    and return value.

    """

    def __init__(self, function, logger):
        """
        :arg function: the function being traced
        :arg logging.Logger logger: the tracing logger

        """
        func_code = function.__code__
        self._func_filename = func_code.co_filename
        self._func_lineno = func_code.co_firstlineno

        self._logger = logger

    @property
    def logger(self):
        """The tracing logger for the function."""
        return self._logger

    def __call__(self, function, args, keywords):
        """Call *function*, tracing its arguments and return value.

        :arg tuple args: the positional arguments for *function*
        :arg dict keywords: the keyword arguments for *function*
        :return:
           the value returned by calling *function* with positional
           arguments *args* and keyword arguments *keywords*

        .. warning::
           This method does **not** perform a level check, and delegates
           *directly* to :meth:`logging.Logger.handle`. The caller is
           expected to perform the level check prior to calling this
           method.

        .. note::
           If the return value of *function* is a `generator iterator
           <https://docs.python.org/3/glossary.html#term-generator-iterator>`_,
           then this method returns *value* wrapped in a
           :class:`_GeneratorIteratorTracingProxy` object to provide the
           ``yield`` and ``StopIteration`` tracing support.

        """
        try:
            logged_user = get_logged_user()
        except Exception:
            logged_user = None

        gid = logged_user.gid if logged_user is not None else ""
        self.logger.log(TRACE, "USER: {} | METHOD: {} | PARAMS: *{} **{}".format(gid, repr(function), args, keywords))
        value = function(*args, **keywords)

        return (_GeneratorIteratorTracingProxy(function, value, self._logger)
                if isgenerator(value) else value)


class _GeneratorIteratorTracingProxy(object):
    """Proxy a generator iterator to capture and trace *YIELD*, *SEND*,
    *THROW*, *CLOSE* and *STOP* events.

    .. note::
       Generator iterators cannot be "rewound." A generator iterator
       that has been exhausted will continue to raise ``StopIteration``
       on all subsequent calls to ``next()``, and Autologging will
       dutifully trace each of those events. This behavior is by design;
       if a program is failing due to an unexpected ``StopIteration``
       exception, then the (traced) program should be able to identify
       when/where the errant ``next()`` call was made.

    """

    #: An easily-queriable marker.
    __autologging_traced__ = True

    def __init__(self, gfunc, giter, logger):
        """
        :arg gfunc:
           the generator function that returned *giter*
        :arg types.GeneratorType iterator:
           the generator iterator returned by *gfunc*
        :arg logging.Logger logger: the tracing logger
        """
        # this is a "fallback" line number for IronPython
        self._gfunc_lineno = gfunc.__code__.co_firstlineno
        self._giter = giter
        self._logger = logger

    #: The wrapped generator iterator.
    @property
    def __wrapped__(self):
        return self._giter

    #: The name of the wrapped generator iterator.
    @property
    def __name__(self):
        return self._giter.__name__

    #: The current line number of the wrapped generator iterator.
    @property
    def _lineno(self):
        # NOTE: IronPython does not track gi_frame.f_lineno correctly (always
        #       reported as 1).
        return (getattr(self._giter.gi_frame, "f_lineno", self._gfunc_lineno)
                if not _is_ironpython else self._gfunc_lineno)

    def __iter__(self):
        """Return a self-reference.

        This method (along with :meth:`__next__`) implements the
        iterator protocol for the proxy object.

        """
        return self

    def _trace(self, message, *message_args):
        giter = self._giter
        self._logger.handle(logging.LogRecord(
            self._logger.name,  # name
            TRACE,  # level
            giter.gi_code.co_filename,  # pathname
            self._lineno,  # lineno
            message,  # msg
            message_args,  # args
            None,  # exc_info
            func=giter.__name__))

    def __next__(self):
        """Attempt to return the next value from the wrapped generator
        iterator.

        If a value is obtained, log the event at :obj:`TRACE` level in a
        "YIELD" record. If the wrapped generator iterator is exhausted,
        log the ``StopIteration`` event (exception) at :obj:`TRACE`
        level in a "STOP" record.

        This method (along with :meth:`__iter__`) implements the
        iterator protocol for the proxy object.

        """
        giter = self._giter
        try:
            value = next(giter)
        except StopIteration:
            self._trace("STOP %r", giter)
            raise
        else:
            self._trace("YIELD %r %r", giter, value)
            return value

    # PYVER: 2.7 compatibility
    next = __next__

    def send(self, value):
        """Send *value* to the wrapped generator iterator, logging
        the event at :obj:`TRACE` level in a "SEND" record.

        """
        giter = self._giter
        self._trace("SEND %r %r", giter, value)
        return giter.send(value)

    def throw(self, exception):
        """Cause the wrapped generator iterator to raise *exception*,
        logging the event at :obj:`TRACE` level in a "THROW" record.

        :arg Exception exception:
           the exception object that the wrapped generator iterator
           should throw

        """
        giter = self._giter
        self._trace("THROW %r %r", giter, exception)
        giter.throw(exception)

    def close(self):
        """Close the wrapped generator iterator, logging the event
        at :obj:`TRACE` level in a "CLOSE" record.

        """
        giter = self._giter
        self._trace("CLOSE %r", giter)
        giter.close()
