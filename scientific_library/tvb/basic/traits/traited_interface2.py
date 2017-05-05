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
Generate Dictionary required by the Framework to generate UI from it.
Returned dictionary will be generated from  traited definition of attributes.


.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart Knock <stuart.knock@gmail.com>
.. moduleauthor:: marmaduke <duke@eml.cc>

"""

import numpy
import json
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.util import get, str_class_name, multiline_math_directives_to_matjax
from tvb.basic.traits.core import KWARG_AVOID_SUBCLASSES, TYPE_REGISTER, KWARG_FILTERS_UI, KWARG_FILTERS_BACKEND
import itree_model as itr
LOG = get_logger(__name__)

INTERFACE_ATTRIBUTES_ONLY = "attributes-only"
INTERFACE_ATTRIBUTES = "attributes"



class TraitedInterfaceGeneratorExperimental(object):
    def __get__(self, inst, ownr):

        obj = inst if inst else ownr
        if not obj.trait.bound:
            return None

        name, label, description, default, required, locked = self.__get_basics(obj)

        mro_type_names = [i.__name__ for i in ownr.mro()]

        if 'Integer' in mro_type_names or 'Float' in mro_type_names:
            type_ = self.__get_wrapped_type(obj)
            range_ = self.__get_range(obj)
            intr = itr.NumericNode(name, type_, default, range_, label, description, required)
        elif 'Array' in mro_type_names:
            range_ = self.__get_range(obj)
            if isinstance(obj.trait.value, numpy.ndarray):
                # Make sure arrays are displayed in a compatible form: [1, 2, 3]
                default = str(obj.trait.value.tolist())
            intr = itr.ArrayNode(name, str(inst.dtype), default, range_, label, description, required)
        elif 'MappedType' in mro_type_names:
            if not ownr._ui_complex_datatype:
                ownr_class = ownr.__class__
            else:
                ownr_class = ownr._ui_complex_datatype
            if 'MetaType' in ownr_class.__name__:
                ownr_class = ownr().__class__
            type_ = ownr_class.__module__ + '.' + ownr_class.__name__

            if not ownr._ui_complex_datatype:
                intr = itr.DatatypeNode(name, type_, label, description, required)
            else:
                attributes = self.__get_entity_attributes(obj)
                intr = itr.ComplexDtypeNode(name, type_, attributes, label, description, required)

            if KWARG_FILTERS_UI in obj.trait.inits.kwd:
                intr.filters_ui = json.dumps([ui_filter.to_dict() for ui_filter in
                                                     obj.trait.inits.kwd[KWARG_FILTERS_UI]])
            if KWARG_FILTERS_BACKEND in obj.trait.inits.kwd:
                intr.conditions = obj.trait.inits.kwd[KWARG_FILTERS_BACKEND]
        elif 'Dict' in mro_type_names and isinstance(default, dict):
            attributes, elementType = self.__prepare_dictionary(default)
            if len(attributes) == 0:
                ## Dictionary without any sub-parameter
                return None
            intr = itr.DictNode(name, elementType, attributes, default)
        elif 'Enumerate' in mro_type_names:
            options = [itr.Option(val, val) for val in obj.trait.options]
            intr = itr.EnumerateNode(name, obj.trait.select_multiple, obj.trait.value, options,
                                     label, description, required)

        ##### TYPE & subclasses specifics ######################################
        elif ('Type' in mro_type_names and obj.__module__ != 'tvb.basic.traits.types_basic'
              or 'Range' in mro_type_names):

            if obj.trait.bound == INTERFACE_ATTRIBUTES_ONLY:
                # We need to do this, to avoid infinite loop on attributes
                # of class Type with no subclasses
                return self.__get_entity_attributes(obj)  # todo edge case if this is not called from __handle_nonmapped_subtypes

            intr = itr.SelectTypeNode(name, obj.trait.select_multiple, [], default, label, description)
            intr.options = self.__handle_nonmapped_subtypes(ownr, obj)

            self.__correct_defaults(intr)
        else:
            type_ = self.__get_wrapped_type(obj)
            intr = itr.LeafNode(name, type_)

        # nonstandard attributes
        self.__fill_noiseconfig(obj, intr)
        intr.locked = locked

        return intr


    @staticmethod
    def __get_basics(obj):
        label = get(obj.trait.inits.kwd, 'label', obj.trait.name)
        if not label:
            label = obj.trait.name
        default = (obj.value or obj.trait.value) if hasattr(obj, 'value') else obj.trait.value
        description = get(obj.trait.inits.kwd, 'doc')
        label = label.capitalize()
        name = obj.trait.name
        locked = obj.trait.inits.kwd.get('locked', False)
        required = obj.trait.inits.kwd.get('required', True)
        return name, label, description, default, required, locked


    @staticmethod
    def __get_range(obj):
        range_value = obj.trait.inits.kwd.get('range', False)
        if range_value:
            minValue = range_value.lo
            maxValue = range_value.hi
            if range_value.step is not None:
                stepValue = range_value.step
            else:
                LOG.debug("Missing Range.step field for attribute %s, we will consider a default." % obj.trait.name)
                stepValue = (range_value.hi - range_value.hi) / 10
            return itr.Range(minValue, maxValue, stepValue)


    @staticmethod
    def __fill_noiseconfig(obj, intr):
        noise_configurable = obj.trait.inits.kwd.get('configurable_noise', None)
        if noise_configurable is not None:
            intr.configurableNoise = noise_configurable


    @staticmethod
    def __get_wrapped_type(obj):
        if isinstance(obj.trait.wraps, tuple):
            return obj.trait.wraps[0].__name__
        else:
            return obj.trait.wraps.__name__


    @staticmethod
    def __get_entity_attributes(obj):
        # Populate Attributes for current entity
        attrs = sorted(obj.trait.values(), key=lambda entity: entity.trait.order_number)
        attrs = [val.interface_experimental for val in attrs if val.trait.order_number >= 0]
        return [attr for attr in attrs if attr is not None]


    def __prepare_dictionary(self, dictionary):
        """
        From base.Dict -> default [isinstance(dict)], prepare an interface specific tree.
        """
        result = []
        element_type = None
        for key in dictionary:
            value = dictionary[key]
            default = str(value)
            mro_type_names = [i.__name__ for i in type(value).mro()]
            if hasattr(value, 'tolist') or 'Array' in mro_type_names:
                type_ = 'array'
                if not hasattr(value, 'tolist'):
                    default = str(value.trait.value)
            else:
                type_ = type(value).__name__

            # assert type uniformity : element_type is type_
            element_type = type_
            if type_ == 'array':
                node = itr.ArrayNode(key, str(value.dtype), default, label=key)
            elif 'Integer' in mro_type_names or 'Float' in mro_type_names:
                node = itr.NumericNode(key, type_, default, label=key)
            else:
                node = itr.LeafNode(key, type_, default, key)
            result.append(node)
        return result, element_type


    @staticmethod
    def __handle_nonmapped_subtypes(ownr, obj):
        """ Populate options for each subtype. This fills in models etc"""
        options = []
        for opt in TYPE_REGISTER.subclasses(ownr, KWARG_AVOID_SUBCLASSES in obj.trait.inits.kwd):
            if hasattr(obj, 'value') and obj.value is not None and isinstance(obj.value, opt):
                ## fill option currently selected with attributes from instance
                opt = obj.value
                opt_class = opt.__class__
            else:
                opt_class = opt

            opt.trait.bound = INTERFACE_ATTRIBUTES_ONLY

            description = multiline_math_directives_to_matjax(opt_class.__doc__)
            name = get(opt, '_ui_name', opt_class.__name__)
            value=str_class_name(opt_class, short_form=True)
            type_=str_class_name(opt_class, short_form=False)
            attributes=opt.interface_experimental
            options.append(itr.TypeNode(name, value, type_, description, attributes))
        return options


    @staticmethod
    def __correct_defaults(intr):
        if intr.default is not None:
            intr.default = intr.default.__class__.__name__
            if intr.default == 'RandomState':
                intr.default = 'RandomStream'



    def __set__(self, inst, val):
        """
        Given a hierarchical dictionary of the kind generated by __get__, with the
        chosen options, we should be able to fully instantiate a class.
        """
        raise NotImplementedError

