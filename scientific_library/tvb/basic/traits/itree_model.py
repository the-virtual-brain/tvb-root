# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
A data model for the input trees.
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

def _repr_help_format_list(lines):
    'Formats a list, trying to keep indentation. It makes repr behave similar to pprint.pformat'
    lines = '\n'.join(str(o) for o in lines)
    lines = lines.splitlines()
    lines = '\n    '.join(lines)
    return lines


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


    def __repr__(self):
        opts = _repr_help_format_list(self.options)
        return ('SelectTypeNode(name=%r, type=%r, default=%r, label=%r,\n'
                '  options=[\n    %s    \n])' % (self.name, self.type, self.default, self.label, opts))


class TypeNode(object):
    '''
    A non-mapped type. Has attributes
    '''
    def __init__(self, name, value, class_, description, attributes):
        self.name = name  # _ui_name
        self.value = value   # short class name
        self.type = class_ # fqn class
        self.description = description # docstring
        self.attributes = attributes

    def __repr__(self):
        attrs = _repr_help_format_list(self.attributes)
        return ('%s(name=%r, class=%r, \n'
                '  attributes=[\n    %s    \n])' %(type(self).__name__, self.name, self.type, attrs))


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

    def __repr__(self):
        return '%s(name=%r, type=%r, label=%r, default=%r)' % (type(self).__name__, self.name, self.type, self.label, self.default)

#class NodeWithInstancesFromDb
class DatatypeNode(object):
    '''
    A dataype node. Has filters.
    '''
    def __init__(self, name, type_, label=None, description=None, required=True, conditions=None, filters_ui=None):
        '''
        type_ may be a FQN or a type
        '''
        if not isinstance(type_, basestring):
            type_ = type_.__module__ + '.' + type_.__name__
        self.name = name
        self.type = type_
        self.label = label or name
        self.description = description or self.label
        self.required = required
        self.conditions = conditions
        self.filters_ui = filters_ui
        self.default = None
        # List of concrete instance gid's. These Options are runtime values, filled from the db after applying filters.
        # This is not a Type level concept, so setting them in a get_input_tree makes no sense.
        self.options = []

    def __repr__(self):
        return '%s(name=%r, type=%r, label=%r, default=%r)' % (
        type(self).__name__, self.name, self.type, self.label, self.default)


class ComplexDtypeNode(object):
    '''
    Usually a cortex. It is not a LeafNode even though similar to a datatypenode
    '''
    def __init__(self, name, type_, attributes, label=None, description=None, required=True, conditions=None, filters_ui=None):
        self.name = name
        self.type = type_
        self.description = description
        self.attributes = attributes
        self.label = label
        self.required = required
        self.conditions = conditions
        self.filters_ui = filters_ui

    def __repr__(self):
        attrs = _repr_help_format_list(self.attributes)
        return ('%s(name=%r, class=%r, \n'
                '  attributes=[\n    %s    \n]\n)' %(type(self).__name__, self.name, self.type, attrs))


class Range(object):
    def __init__(self, min_, max_, step):
        self.min = min_
        self.max = max_
        self.step = step

    def __repr__(self):
        return 'Range(%d, %d, %d)'% (self.min, self.max, self.step)


class NumericNode(LeafNode):
    def __init__(self, name, type_, default=0, range_=None,
                 label=None, description=None, required=True):
        LeafNode.__init__(self, name, type_, default, label, description, required)
        self.range = range_

    def __repr__(self):
        return '%s(name=%r, type=%r, label=%r, default=%r, range=%r)' % (
            type(self).__name__, self.name, self.type, self.label, self.default, self.range)


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

    def __repr__(self):
        attrs = _repr_help_format_list(self.attributes)
        return ('DictNode(name=%r, type=%r, label=%r, default=%r)\n'
                '    attributes=[\n    %s    \n]\n'
                % (self.name, self.type, self.label, self.default, attrs))


class EnumerateNode(object):
    def __init__(self, name, select_multiple, default, options,
                 label=None, description=None, required=True):
        type_ = 'selectMultiple' if select_multiple else 'select'
        self.name = name
        self.type = type_
        self.label = label or name
        self.description = description or self.label
        self.required = required
        self.default = default
        self.options = options  # of type Option. These options are derived from the Type traits.

    def __repr__(self):
        return '%s(name=%r, type=%r, label=%r, default=%r, options=%r)' % (
            type(self).__name__, self.name, self.type, self.label, self.default, self.options)


class Option(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return repr((self.name, self.value))

