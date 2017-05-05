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
Traits metadata module
======================

introduction
------------

The problem is that various classes have members whose properties we want to
annotate in a way that does not affect the execution of the class algorithms
but that exposes an API to pull out this info for a user interface and
documentation. It should be readable for the programmers as well. We refer to
an annotated/propertied member as a trait.

Consider the following class:

    >>> class Point(object):
    ...     x = 0.0
    ...     y = 0.0

If you want to present this data to the user, you may want to label each
attribute of the Point with more than just 'x' or 'y', or you want to check
that if the user creates a Point, the values have the right type (here, we
assume the type of 0.0 means floating point), and if you want to store the
instances of Point in a data base, each one needs a unique id. These pieces of
information need to be put somewhere, and the best place to put it is in the
attribute declaration, for example,

    >>> class Point(Type):
    ...     x = Float(0.0, doc='horizontal position')
    ...     y = Float(0.0, doc='vertical position')

Here, the type of each attribute is explicit, and a documentation string
is provided, *all in the same attribute declaration*, and this co-location
of information makes it easier to maintain, and elsewhere in the program,
this information can be accessed programmatically to generate user interfaces,
serialize data, validate data, etc. Additionally, all classes' docstrings
will have Traits information automatically added:

    >>> print Point.__doc__
    Traited class Point
        x: horizontal position, type float
        y: vertical position, type float

The Traits package allows us to write classes like the one above by using a
metaclass to customize the class creation. In brief, when Python reads a class
definition the Trait metaclass has access to all the attribute declarations,
and it will place a dictionary of all the data provided in the declarations on
the class.  When the classes are instantiated and used, the Trait attributes
behave exactly like their values, such that class methods do not need to know
they are accessing Trait attributes. For more details, see `traits.core`.


package modules
---------------

    core            Provides the core of Traits, a metaclass and class that
                    implement the reflection/annotation that Traits allows.

    types_basic     Provides basic predefined types as well as the Type that
                    should be inherited from when making new Types

    traited_interface  Provides recursive interfaces classes that are data
                    descriptors, i.e. when placed on classes, they compute
                    the full description of the class and it's Traited
                    attributes.

    util            Utility functions and classes.


automatic class documentation
`````````````````````````````

Classes that subclass Type (or subclasses of MetaType) have their docstrings
automatically augmented by the repr() of each attribute which is a Type. This
allows one to find the trait documentation both in SPhinx and at the console.

References
----------

Suggested reading is `Unifying Types and Classes`_ and th `Python data model`_.

.. _Unifying Types and Classes:
    http://www.python.org/download/releases/2.2.3/descrintro/
.. _Python data model: http://docs.python.org/reference/datamodel.html

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import core
import traited_interface, traited_interface2
from tvb.basic.profile import TvbProfile

# Add interfaces based on configured parameter on classes
setattr(core.Type, TvbProfile.current.TRAITS_CONFIGURATION.interface_method_name,
        traited_interface.TraitedInterfaceGenerator())
setattr(core.Type, 'interface_experimental',
        traited_interface2.TraitedInterfaceGeneratorExperimental())



