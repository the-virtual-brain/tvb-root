# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import json
import uuid
from collections import namedtuple

import numpy
from tvb.basic.neotraits.api import List, Attr
from tvb.basic.neotraits.ex import TraitError
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom.h5 import REGISTRY
# TODO: remove dependency
from tvb.core.neotraits.db import HasTraitsIndex
from tvb.core.neotraits.view_model import DataTypeGidAttr

# This setting is injected.
# The pattern might be confusing, but it is an interesting alternative to
# universal tvbprofile imports

jinja_env = None


class Field(object):
    template = None

    def __init__(self, name, disabled=False, required=False, label='', doc='', default=None):
        # type: (str, bool, bool, str, str, object) -> None
        self.name = name
        self.disabled = disabled
        self.required = required
        self.label = label
        self.doc = doc
        self.label_classes = []
        if required:
            self.label_classes.append('field-mandatory')

        # keeps the deserialized data
        self.data = None
        # keeps user input, even if wrong, we have to redisplay it
        self.unvalidated_data = default
        self.errors = []

    def fill_from_post(self, post_data):
        """ deserialize form a post dictionary """
        self.unvalidated_data = post_data.get(self.name)
        try:
            self._from_post()
        except ValueError as ex:
            self.errors.append(ex)

    def validate(self):
        """ validation besides the deserialization from post"""
        return not self.errors

    def _from_post(self):
        if self.required and (self.unvalidated_data is None or len(self.unvalidated_data.strip()) == 0):
            raise ValueError('Field required')
        self.data = self.unvalidated_data

    @property
    def value(self):
        if self.data is not None:
            return self.data
        return self.unvalidated_data

    def __repr__(self):
        return '<{}>(name={})'.format(type(self).__name__, self.name)

    def __str__(self):
        return jinja_env.get_template(self.template).render(field=self)


class TraitField(Field):
    def __init__(self, trait_attribute, name=None, disabled=False):
        # type: (Attr, str, bool) -> None
        self.trait_attribute = trait_attribute  # type: Attr
        name = name or trait_attribute.field_name
        label = trait_attribute.label or name

        super(TraitField, self).__init__(
            name,
            disabled,
            trait_attribute.required,
            label,
            trait_attribute.doc,
            trait_attribute.default
        )

    def from_trait(self, trait, f_name):
        self.data = getattr(trait, f_name)


TEMPORARY_PREFIX = ".tmp"


class TraitUploadField(TraitField):
    template = 'form_fields/upload_field.html'

    def __init__(self, traited_attribute, required_type, name, disabled=False):
        super(TraitUploadField, self).__init__(traited_attribute, name, disabled)
        self.required_type = required_type


class TraitDataTypeSelectField(TraitField):
    template = 'form_fields/datatype_select_field.html'
    missing_value = 'explicit-None-value'

    def __init__(self, trait_attribute, name=None, conditions=None,
                 draw_dynamic_conditions_buttons=True, has_all_option=False,
                 show_only_all_option=False):
        super(TraitDataTypeSelectField, self).__init__(trait_attribute, name)

        if issubclass(type(trait_attribute), DataTypeGidAttr):
            type_to_query = trait_attribute.linked_datatype
        else:
            type_to_query = trait_attribute.field_type

        if issubclass(type_to_query, HasTraitsIndex):
            self.datatype_index = type_to_query
        else:
            self.datatype_index = REGISTRY.get_index_for_datatype(type_to_query)
        self.conditions = conditions
        self.draw_dynamic_conditions_buttons = draw_dynamic_conditions_buttons
        self.has_all_option = has_all_option
        self.show_only_all_option = show_only_all_option
        self.datatype_options = []

    def from_trait(self, trait, f_name):
        if hasattr(trait, f_name):
            self.data = getattr(trait, f_name)
            if isinstance(self.data, uuid.UUID):
                self.data = self.data.hex

    @property
    def get_dynamic_filters(self):
        return FilterChain().get_filters_for_type(self.datatype_index)

    @property
    def get_form_filters(self):
        return self.conditions

    def options(self):
        if not self.required:
            choice = None
            yield Option(
                id='{}_{}'.format(self.name, None),
                value=self.missing_value,
                label=str(choice).title(),
                checked=self.data is None
            )

        if not self.show_only_all_option:
            for i, dt_opt in enumerate(self.datatype_options):
                yield Option(
                    id='{}_{}'.format(self.name, i),
                    value=dt_opt[0][2],
                    label=dt_opt[1],
                    checked=self.data == dt_opt[0][2]
                )

        if self.has_all_option:

            all_values = ''
            for fdt in self.datatype_options:
                all_values += str(fdt[0][2]) + ','

            choice = "All"
            yield Option(
                id='{}_{}'.format(self.name, choice),
                value=all_values[:-1],
                label=choice,
                checked=self.data is choice
            )

    def _from_post(self):
        if self.unvalidated_data == self.missing_value:
            self.unvalidated_data = None

        if self.required and not self.unvalidated_data:
            raise ValueError('Field required')

        # TODO: ensure is in choices
        try:
            if self.unvalidated_data is None:
                self.data = None
            else:
                self.data = uuid.UUID(self.unvalidated_data)
        except ValueError:
            raise ValueError('The chosen entity does not have a proper GID')


class StrField(TraitField):
    template = 'form_fields/str_field.html'


class BoolField(TraitField):
    template = 'form_fields/bool_field.html'

    def _from_post(self):
        self.data = self.unvalidated_data is not None


class IntField(TraitField):
    template = 'form_fields/number_field.html'
    min = None
    max = None

    def _from_post(self):
        super(IntField, self)._from_post()
        if len(self.unvalidated_data.strip()) == 0:
            self.data = None
        else:
            self.data = int(self.unvalidated_data)


class FloatField(TraitField):
    template = 'form_fields/number_field.html'
    min = None
    max = None
    step = 'any'

    def _from_post(self):
        super(FloatField, self)._from_post()
        if len(self.unvalidated_data.strip()) == 0:
            self.data = None
        else:
            self.data = float(self.unvalidated_data)


class ArrayField(TraitField):
    template = 'form_fields/str_field.html'

    def _from_post(self):
        super(ArrayField, self)._from_post()
        self.data = None
        if len(self.unvalidated_data.strip()) == 0:
            self.data = None
        else:
            data = json.loads(self.unvalidated_data)
            self.data = numpy.array(data, dtype=self.trait_attribute.dtype)

    @property
    def value(self):
        if self.data is None:
            # this None means self.data is missing, either not set or unset cause of validation error
            return self.unvalidated_data
        try:
            if self.data.size > 100:
                data_to_display = self.data[:100]
            else:
                data_to_display = self.data
            return json.dumps(data_to_display.tolist())
        except (TypeError, ValueError):
            return self.unvalidated_data


Option = namedtuple('Option', ['id', 'value', 'label', 'checked'])


class SelectField(TraitField):
    template = 'form_fields/radio_field.html'
    missing_value = 'explicit-None-value'
    subform_prefix = 'subform_'

    def _prepare_template(self, choices):
        if len(choices) > 4:
            self.template = 'form_fields/select_field.html'

    def __init__(self, trait_attribute, name=None, disabled=False, choices=None, display_none_choice=True,
                 subform=None, display_subform=True):
        super(SelectField, self).__init__(trait_attribute, name, disabled)
        if choices:
            self.choices = choices
        else:
            self.choices = {choice: choice for choice in trait_attribute.choices}
        if not self.choices:
            raise ValueError('no choices for field')
        self.display_none_choice = display_none_choice
        self.subform_field = None
        if subform:
            self.subform_field = FormField(subform, self.subform_prefix + self.name)
            self.display_subform = display_subform
        self._prepare_template(self.choices)

    @property
    def value(self):
        if self.data is None and not self.trait_attribute.required:
            return self.data
        return super(SelectField, self).value

    def options(self):
        """ to be used from template, assumes self.data is set """
        if self.display_none_choice:
            if not self.trait_attribute.required:
                choice = None
                yield Option(
                    id='{}_{}'.format(self.name, None),
                    value=self.missing_value,
                    label=str(choice).title(),
                    checked=self.data is None
                )

        for i, choice in enumerate(self.choices):
            yield Option(
                id='{}_{}'.format(self.name, i),
                value=choice,
                label=str(choice).title(),
                checked=self.value == self.choices.get(choice)
            )

    def _from_post(self):
        super(SelectField, self)._from_post()

        if self.data != self.missing_value and self.choices.get(self.data) is None:
            raise ValueError("the entered value is not among the choices for this field!")

        self.data = self.choices.get(self.data)


class MultiSelectField(TraitField):
    template = 'form_fields/checkbox_field.html'

    def __init__(self, trait_attribute, name=None, disabled=False):
        super(MultiSelectField, self).__init__(trait_attribute, name, disabled)
        if not isinstance(trait_attribute, List):
            raise NotImplementedError('only List in multi select for now')

    def options(self):
        """ to be used from template, assumes self.data is set """
        for i, choice in enumerate(self.trait_attribute.element_choices):
            yield Option(
                id='{}_{}'.format(self.name, i),
                value=choice,
                label=str(choice).title(),
                checked=self.data is not None and choice in self.data
            )

    def _from_post(self):
        if self.unvalidated_data is None:
            if self.required:
                raise ValueError('Field required')
            else:
                return None

        if not isinstance(self.unvalidated_data, list):
            selected = [self.unvalidated_data]
        else:
            selected = self.unvalidated_data

        data = []  # don't mutate self.data until we know all values converted ok

        for s in selected:
            converted_s = self.trait_attribute.element_type(s)
            if converted_s not in self.trait_attribute.element_choices:
                raise ValueError('must be one of {}'.format(self.trait_attribute.element_choices))
            data.append(converted_s)

        self.data = data


class HiddenField(TraitField):
    template = 'form_fields/hidden_field.html'

    def __init__(self, trait_attribute, name=None, disabled=False):
        super(HiddenField, self).__init__(trait_attribute, name, disabled)
        self.trait_attribute.label = ''


class FormField(Field):
    template = 'form_fields/form_field.html'

    def __init__(self, form_class, name, label='', doc=''):
        super(FormField, self).__init__(name, False, False, label, doc)
        self.form = form_class()

    def fill_from_post(self, post_data):
        self.form.fill_from_post(post_data)
        self.errors = self.form.errors

    def validate(self):
        return self.form.validate()

    def __str__(self):
        return jinja_env.get_template(self.template).render(adapter_form=self.form)


class Form(object):

    def __init__(self):
        self.errors = []

    def get_subform_key(self):
        """
        If the current form can be used as subform, this method should return the proper value from SubformsEnum.
        """
        raise NotImplementedError

    @property
    def fields(self):
        for field in self.__dict__.values():
            if isinstance(field, Field):
                yield field

    @property
    def trait_fields(self):
        for field in self.__dict__.values():
            if isinstance(field, TraitField):
                yield field

    def validate(self):
        valid = True
        for field in self.fields:
            if not field.validate():
                valid = False
        return valid

    def get_errors_dict(self):
        result = {}
        for field in self.fields:
            if not field.validate():
                result[field.name] = field.errors[0]
        return result

    def fill_from_trait(self, trait):
        """
        Sets data for all traited fields from a trait instance.
        Note that FormFields are not TraitFields, so this does not work recursively
        Override to fill in sub-forms
        """
        for field in self.trait_fields:
            f_name = field.trait_attribute.field_name
            if f_name is None:
                # skipp attribute that does not seem to belong to a traited type
                continue
            field.from_trait(trait, f_name)

    def fill_trait(self, datatype):
        """
        Copies the value of the TraitFields to the corresponding Attr-ibutes of the given trait instance
        Note that FormFields are not TraitFields, so this does not work recursively
        Override to fill in sub-forms
        """
        for field in self.trait_fields:
            f_name = field.trait_attribute.field_name
            if f_name is None:
                # skipp attribute that does not seem to belong to a traited type
                continue
            try:
                setattr(datatype, f_name, field.data)
            except TraitError as ex:
                # as field.data is clearly tainted set it to None so that field.unvalidated_data
                # will render and the user can fix the typo's
                field.data = None
                field.errors.append(ex)
                raise

    def fill_trait_partially(self, datatype, fields=None):
        for field in self.trait_fields:
            f_name = field.trait_attribute.field_name
            if f_name is None or \
                    fields is not None and f_name not in fields:
                # skipp attribute that does not seem to belong to a traited type
                continue
            try:
                setattr(datatype, f_name, field.data)
            except TraitError as ex:
                # as field.data is clearly tainted set it to None so that field.unvalidated_data
                # will render and the user can fix the typo's
                field.data = None
                field.errors.append(ex)
                raise

    def fill_from_post(self, form_data):
        for field in self.fields:
            field.fill_from_post(form_data)

    def fill_from_single_post_param(self, **param):
        param_key = list(param)[0]
        field = getattr(self, param_key)
        field.fill_from_post(param)
