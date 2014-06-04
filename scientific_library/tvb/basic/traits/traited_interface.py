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
from tvb.basic.traits.core import KWARG_AVOID_SUBCLASSES, TYPE_REGISTER, KWARG_FILTERS_UI

LOG = get_logger(__name__)

INTERFACE_ATTRIBUTES_ONLY = "attributes-only"
INTERFACE_ATTRIBUTES = "attributes"



class TraitedInterfaceGenerator(object):
    """
    Bases class for interface reading and dumping. As a data descriptor, when 
    it is an attribute of the class it will compute a dictionary and return it.
    """


    def __get__(self, inst, ownr):

        obj = inst if inst else ownr
        if not obj.trait.bound:
            return {}

        label = get(obj.trait.inits.kwd, 'label', obj.trait.name)
        if not label:
            label = obj.trait.name
        intr = {'default': (obj.value or obj.trait.value) if hasattr(obj, 'value') else obj.trait.value,
                'description': get(obj.trait.inits.kwd, 'doc'),
                'label': self.label(label),
                'name': obj.trait.name,
                'locked': obj.trait.inits.kwd.get('locked', False),
                'required': obj.trait.inits.kwd.get('required', True)}

        range_value = obj.trait.inits.kwd.get('range', False)
        if range_value:
            intr['minValue'] = range_value.lo
            intr['maxValue'] = range_value.hi
            if range_value.step is not None:
                intr['stepValue'] = range_value.step
            else:
                LOG.debug("Missing Range.step field for attribute %s, we will consider a default." % obj.trait.name)
                intr['stepValue'] = (range_value.hi - range_value.hi) / 10
            
        noise_configurable = obj.trait.inits.kwd.get('configurable_noise', None)
        if noise_configurable is not None:
            intr['configurableNoise'] = noise_configurable

        if KWARG_FILTERS_UI in obj.trait.inits.kwd:
            intr[KWARG_FILTERS_UI] = json.dumps([ui_filter.to_dict() for ui_filter in
                                                 obj.trait.inits.kwd[KWARG_FILTERS_UI]])

        if hasattr(obj, 'dtype'):
            intr['elementType'] = getattr(obj, 'dtype')

        if get(obj.trait, 'wraps', False):
            if isinstance(obj.trait.wraps, tuple):
                intr['type'] = str(obj.trait.wraps[0].__name__)
            else:
                intr['type'] = str(obj.trait.wraps.__name__)

            if intr['type'] == 'dict' and isinstance(intr['default'], dict):
                intr['attributes'], intr['elementType'] = self.__prepare_dictionary(intr['default'])
                if len(intr['attributes']) < 1:
                    ## Dictionary without any sub-parameter
                    return {}

        ##### ARRAY specific processing ########################################
        if 'Array' in [str(i.__name__) for i in ownr.mro()]:
            intr['type'] = 'array'
            intr['elementType'] = str(inst.dtype)
            intr['quantifier'] = 'manual'
            if obj.trait.value is not None and isinstance(obj.trait.value, numpy.ndarray):
                # Make sure arrays are displayed in a compatible form: [1, 2, 3]
                intr['default'] = str(obj.trait.value.tolist())

        ##### TYPE & subclasses specifics ######################################
        elif ('Type' in [str(i.__name__) for i in ownr.mro()]
              and (obj.__module__ != 'tvb.basic.traits.types_basic'
                   or 'Range' in [str(i.__name__) for i in ownr.mro()])
              or 'Enumerate' in [str(i.__name__) for i in ownr.mro()]):

            # Populate Attributes for current entity
            attrs = sorted(getattr(obj, 'trait').values(), key=lambda entity: entity.trait.order_number)
            attrs = [val.interface for val in attrs if val.trait.order_number >= 0]
            attrs = [attr for attr in attrs if attr is not None and len(attr) > 0]
            intr['attributes'] = attrs

            if obj.trait.bound == INTERFACE_ATTRIBUTES_ONLY:
                # We need to do this, to avoid infinite loop on attributes 
                # of class Type with no subclasses
                return intr

            if obj.trait.select_multiple:
                intr['type'] = 'selectMultiple'
            else:
                intr['type'] = 'select'

            ##### MAPPED_TYPE specifics ############################################
            if 'MappedType' in [str(i.__name__) for i in ownr.mro()]:
                intr['datatype'] = True
                #### For simple DataTypes, cut options and attributes
                intr['options'] = []
                if not ownr._ui_complex_datatype:
                    intr['attributes'] = []
                    ownr_class = ownr.__class__
                else:
                    ownr_class = ownr._ui_complex_datatype
                if 'MetaType' in ownr_class.__name__:
                    ownr_class = ownr().__class__
                intr['type'] = ownr_class.__module__ + '.' + ownr_class.__name__

            else:
        ##### TYPE (not MAPPED_TYPE) again ####################################
                intr['attributes'] = []
                # Build options list
                intr['options'] = []
                if 'Enumerate' in obj.__class__.__name__:
                    for val in obj.trait.options:
                        intr['options'].append({'name': val,
                                                'value': val})
                    intr['default'] = obj.trait.value
                    return intr
                else:
                    for opt in TYPE_REGISTER.subclasses(ownr, KWARG_AVOID_SUBCLASSES in obj.trait.inits.kwd):
                        if hasattr(obj, 'value') and obj.value is not None and isinstance(obj.value, opt):
                            ## fill option currently selected with attributes from instance
                            opt = obj.value
                            opt_class = opt.__class__
                        else:
                            opt_class = opt
                        opt.trait.bound = INTERFACE_ATTRIBUTES_ONLY

                        description = multiline_math_directives_to_matjax(opt_class.__doc__)

                        intr['options'].append({'name': get(opt, '_ui_name', opt_class.__name__),
                                                'value': str_class_name(opt_class, short_form=True),
                                                'class': str_class_name(opt_class, short_form=False),
                                                'description': description,
                                                'attributes': opt.interface['attributes']})

            if intr['default'] is not None and intr['default'].__class__:
                intr['default'] = str(intr['default'].__class__.__name__)
                if intr['default'] == 'RandomState':
                    intr['default'] = 'RandomStream'
            else:
                intr['default'] = None

        return intr


    def __prepare_dictionary(self, dictionary):
        """
        From base.Dict -> default [isinstance(dict)], prepare an interface specific tree.
        """
        result = []
        element_type = None
        for key in dictionary:
            entry = {}
            value = dictionary[key]
            entry['label'] = key
            entry['name'] = key
            if type(value).__name__ == 'dict':
                entry['attributes'], entry['elementType'] = self.__prepare_dictionary(value)
                value = ''
            entry['default'] = str(value)

            if hasattr(value, 'tolist') or 'Array' in [str(i.__name__) for i in type(value).mro()]:
                entry['type'] = 'array'
                if not hasattr(value, 'tolist'):
                    entry['default'] = str(value.trait.value)
            else:
                entry['type'] = type(value).__name__

            element_type = entry['type']
            result.append(entry)
        return result, element_type


    def __set__(self, inst, val):
        """
        Given a hierarchical dictionary of the kind generated by __get__, with the 
        chosen options, we should be able to fully instantiate a class.
        """
        raise NotImplementedError


    @staticmethod
    def label(text):
        """
        Create suitable UI label given text.
        Enforce starts with upper-case.
        """
        return text[0].upper() + text[1:]
    

