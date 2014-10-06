# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
All the little functions that make life nicer in the Traits package.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: marmaduke <duke@eml.cc>
"""

import numpy
import collections
import inspect
from tvb.basic.profile import TvbProfile


# returns true if key is, by convention, public
ispublic = lambda key: key[0] is not '_'


def str_class_name(thing, short_form=False):
    """
    A helper function that tries to generate an informative name for its
    argument: when passed a class, return its name, when passed an object
    return a string representation of that value.
    """
    # if thing is a class, it has attribute __name__
    if hasattr(thing, '__name__'):
        cls = thing
        if short_form:
            return cls.__name__
        return cls.__module__ + '.' + cls.__name__
    else:
        # otherwise, it's an object and we return its __str__
        return str(thing)



def get(obj, key, default=None):
    """
    get() is a general function allowing us to ignore whether we are
    getting from a dictionary or object. If obj is a dictionary, we
    return the value corresponding to key, otherwise we return the
    attribute on obj corresponding to key. In both cases, if key
    does not exist, default is returned.
    """
    if type(obj) is dict:
        return obj.get(key, default)
    else:
        return getattr(obj, key) if hasattr(obj, key) else default



def log_debug_array(log, array, array_name, owner=""):
    """
    Simple access to debugging info on an array.
    """
    if TvbProfile.current.TRAITS_CONFIGURATION.use_storage:
        return
        # Hide this logs in web-mode, with storage, because we have multiple storage exceptions

    if owner != "":
        name = ".".join((owner, array_name))
    else:
        name = array_name

    if array is not None and hasattr(array, 'shape'):
        shape = str(array.shape)
        dtype = str(array.dtype)
        has_nan = str(numpy.isnan(array).any())
        array_max = str(array.max())
        array_min = str(array.min())
        log.debug("%s shape: %s" % (name, shape))
        log.debug("%s dtype: %s" % (name, dtype))
        log.debug("%s has NaN: %s" % (name, has_nan))
        log.debug("%s maximum: %s" % (name, array_max))
        log.debug("%s minimum: %s" % (name, array_min))
    else:
        log.debug("%s is None or not Array" % name)



Args = collections.namedtuple('Args', 'pos kwd')



class TypeRegister(list):
    """
    TypeRegister is a smart list that can be queried to obtain selections of the
    classes inheriting from Traits classes.
    """


    def subclasses(self, obj, avoid_subclasses=False):
        """
        The subclasses method takes a class (or given instance object, will use
        the class of the instance), and returns a list of all options known to
        this TypeRegister that are direct subclasses of the class or have the
        class in their base class list.
        :param obj: Class or instance
        :param avoid_subclasses: When specified, subclasses are not retrieved, only current class.
        """

        cls = obj if inspect.isclass(obj) else obj.__class__

        if avoid_subclasses:
            return [cls]

        if hasattr(cls, '_base_classes'):
            bases = cls._base_classes
        else:
            bases = []

        sublcasses = [opt for opt in self if ((issubclass(opt, cls) or cls in opt.__bases__)
                                              and not inspect.isabstract(opt) and opt.__name__ not in bases)]
        return sublcasses


def multiline_math_directives_to_matjax(doc):
    """
    Looks for multi-line sphinx math directives in the given rst string
    It converts them in html text that will be interpreted by mathjax
    The parsing is simplistic, not a rst parser.
    Wraps .. math :: body in \[\begin{split}\end{split}\]
    """

    # doc = text | math
    BEGIN = r'\[\begin{split}'
    END = r'\end{split}\]'

    in_math = False  # 2 state parser
    out_lines = []
    indent = ''

    for line in doc.splitlines():
        if not in_math:
            # math = indent directive math_body
            indent, sep, _ = line.partition('.. math::')
            if sep:
                out_lines.append(BEGIN)
                in_math = True
            else:
                out_lines.append(line)
        else:
            # math body is at least 1 space more indented than the directive, but we tolerate empty lines
            if line.startswith(indent + ' ') or line.strip() == '':
                out_lines.append(line)
            else:
                # this line is not properly indented, math block is over
                out_lines.append(END)
                out_lines.append(line)
                in_math = False

    if in_math:
        # close math tag
        out_lines.append(END)

    return '\n'.join(out_lines)