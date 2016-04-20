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
    def __init__(self, name, select_multiple, options, default=None, label=None, description=None):
        self.name = name
        self.label = label or name
        self.description = description or self.label
        self.type = 'selectMultiple' if select_multiple else 'select'
        self.options = options # of type TypeNode
        if default is None and len(options):
            self.default = options[0].name
        else:
            self.default = default

class TypeNode(object):
    '''
    A non-mapped type. Has attributes
    '''
    def __init__(self, name, value, class_, description, attributes):
        self.name = name  # _ui_name
        self.value = value   # short class name
        self.class_ = class_ # fqn class
        self.description = description # docstring
        self.attributes = attributes


class LeafNode(object):
    '''
    Node has no descendants
    '''
    def __init__(self, name, type_, default=None, label=None, description=None, required=True):
        self.name = name
        self.type = type_
        self.label = label or name
        self.description = description or self.label
        self.required = required
        self.default = default


class DatatypeNode(LeafNode):
    '''
    A dataype node. Has filters.
    '''
    def __init__(self, name, type_, label=None, description=None, required=True, conditions=None, filters_ui=None):
        '''
        type_ may be a FQN or a type
        '''
        if not isinstance(type_, basestring):
            type_ = type_.__module__ + '.' + type_.__name__
        LeafNode.__init__(self, name, type_, None, label, description, required)
        self.conditions = conditions
        self.filters_ui = filters_ui


class ComplexDtypeNode(TypeNode):
    '''
    Usually a cortex. It is not a LeafNode even though similar to a datatypenode
    '''
    def __init__(self, name, type_, attributes, label=None, description=None, required=True, conditions=None, filters_ui=None):
        TypeNode.__init__(self, name, 'value', type_, description, attributes)
        self.label = label
        self.required = required
        self.conditions = conditions
        self.filters_ui = filters_ui


class Range(object):
    def __init__(self, min_, max_, step):
        self.min = min_
        self.max = max_
        self.step = step


class NumericNode(LeafNode):
    def __init__(self, name, type_, default=0, range_=None,
                 label=None, description=None, required=True):
        LeafNode.__init__(self, name, type_, default, label, description, required)
        self.range = range_



class ArrayNode(NumericNode):
    def __init__(self, name, type_, default='[]', range_=None,
                 label=None, description=None, required=True):
        NumericNode.__init__(self, name, type_, default, range_, label, description, required)


class DictNode(object):
    def __init__(self, name, type_, attributes, default):
        self.name = name
        self.label = name
        self.type = type_  # todo: is it useful to keep the concrete leaf node type?
        self.default = default
        self.attributes = attributes # type Leafnode


# todo is this really a leaf? it has options and select type. In a sense it is a leaf, as the options are trivial
class EnumerateNode(LeafNode):
    def __init__(self, name, select_multiple, default, options,
                 label=None, description=None, required=True):
        type_ = 'selectMultiple' if select_multiple else 'select'
        LeafNode.__init__(self, name, type_, default, label, description, required)
        self.options = options  # of type EnumeratedOption


class EnumeratedOption(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
