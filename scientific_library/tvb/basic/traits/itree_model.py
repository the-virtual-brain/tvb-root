# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
A data model for the input trees.
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""


class SelectTypeNode(object):
    '''
    This node represents a collection of non mapped Types.
    Examples Model collection, Integrator collection
    '''
    def __init__(self, name, options, label=None, description=None):
        self.name = name
        self.label = label or name
        self.description = description or self.label
        self.options = options # of type TypeNode


class TypeNode(object):
    '''
    A non-mapped type. Has attributes
    '''
    def __init__(self, name, attributes,  value, class_, description):
        self.name = name  # _ui_name
        self.value = value   # short class name
        self.class_ = class_ # fqn class
        self.description = description # docstring
        self.attributes = attributes


class LeafNode(object):
    '''
    Node has no descendants
    '''
    def __init__(self, name, type_, label=None, description=None, required=True, locked=False):
        self.name = name
        self.type = type_
        self.label = label or name
        self.description = description or self.label
        self.required = required
        self.locked = locked


class DatatypeNode(LeafNode):
    '''
    A dataype node. Has filters.
    '''
    def __init__(self, name, type_, label=None, description=None, required=True, locked=False):
        LeafNode.__init__(self, name, type_, label, description, required, locked)
        self.conditions = None
        self.filters_ui = None

# class ComplexDtypeNode(TypeNode): # cortex
#     pass

class Range(object):
    def __init__(self, min_, max_, step):
        self.min = min_
        self.max = max_
        self.step = step


class NumericNode(LeafNode):
    def __init__(self, name, type_, default, range_=None,
                 label=None, description=None, required=True, locked=False):
        LeafNode.__init__(self, name, type_, label, description, required, locked)
        self.default = default
        self.range = range_


class ArrayNode(LeafNode):
    def __init__(self, name, type_, default, range_=None, quantifier='manual',
                 label=None, description=None, required=True, locked=False):
        LeafNode.__init__(self, name, 'array', label, description, required, locked)

        self.default = default
        self.range = range_
        self.elementType = type_
        self.quantifier = quantifier


class DictNode(object):
    def __init__(self):
        self.name = None
        self.label = None
        self.default = None
        self.type = None
        self.attributes = [] # type Leafnode
        self.elementType = None


class EnumerateNode(object):
    def __init__(self, name, options):
        self.name = name
        self.options = options  # of type EnumeratedOption


class EnumeratedOption(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
