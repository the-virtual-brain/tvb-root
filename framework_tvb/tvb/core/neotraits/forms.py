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

import os
import json
import uuid
from collections import namedtuple
from datetime import datetime
import numpy
from tvb.core import utils
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.storage import dao
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom.h5 import REGISTRY
from tvb.basic.neotraits.ex import TraitError
from tvb.basic.neotraits.api import List, Attr
from tvb.core.neotraits.db import HasTraitsIndex
from tvb.core.neotraits.view_model import DataTypeGidAttr

# This setting is injected.
# The pattern might be confusing, but it is an interesting alternative to
# universal tvbprofile imports

jinja_env = None


def prepare_prefixed_name_for_field(prefix, name):
    if prefix != "":
        return '{}_{}'.format(prefix, name)
    else:
        return name


class Field(object):
    template = None

    def __init__(self, form, name, disabled=False, required=False, label='', doc='', default=None):
        # type: (Form, str, bool, bool, str, str, object) -> None
        self.owner = form
        self.name = prepare_prefixed_name_for_field(form.prefix, name)
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
        # todo
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
        if self.required and self.unvalidated_data is None:
            raise ValueError('Field required')
        self.data = self.unvalidated_data

    @property
    def value(self):
        if str(self.data) == self.unvalidated_data:
            return self.data
        return self.data or self.unvalidated_data

    def __repr__(self):
        return '<{}>(name={})'.format(type(self).__name__, self.name)

    def __str__(self):
        return jinja_env.get_template(self.template).render(field=self)


class SimpleBoolField(Field):
    template = 'form_fields/bool_field.html'

    def _from_post(self):
        if self.unvalidated_data is None:
            self.data = False
        self.data = bool(self.unvalidated_data)


class SimpleStrField(Field):
    template = 'form_fields/str_field.html'

    def _from_post(self):
        if self.required and (self.unvalidated_data is None or self.unvalidated_data.strip() == ''):
            raise ValueError('Field required')
        self.data = self.unvalidated_data


class SimpleHiddenField(Field):
    template = 'form_fields/hidden_field.html'


class SimpleIntField(Field):
    template = 'form_fields/number_field.html'
    min = None
    max = None

    def _from_post(self):
        super(SimpleIntField, self)._from_post()
        if self.unvalidated_data is None or self.unvalidated_data.strip() == '':
            if self.required:
                raise ValueError('Field required')
            self.data = None
        else:
            self.data = int(self.unvalidated_data)


class SimpleFloatField(Field):
    template = 'form_fields/number_field.html'
    input_type = "number"
    min = None
    max = None
    step = 'any'

    def _from_post(self):
        super(SimpleFloatField, self)._from_post()
        if self.unvalidated_data is None or self.unvalidated_data.strip() == '':
            if self.required:
                raise ValueError('Field required')
            self.data = None
        else:
            self.data = float(self.unvalidated_data)


class SimpleSelectField(Field):
    template = 'form_fields/radio_field.html'
    missing_value = 'explicit-None-value'

    def __init__(self, choices, form, name=None, disabled=False, required=False, label='', doc='', default=None,
                 include_none=True):
        super(SimpleSelectField, self).__init__(form, name, disabled, required, label, doc, default)
        self.choices = choices
        self.include_none = include_none

    def options(self):
        """ to be used from template, assumes self.data is set """
        if not self.required and self.include_none:
            choice = None
            yield Option(
                id='{}_{}'.format(self.name, None),
                value=self.missing_value,
                label=str(choice).title(),
                checked=self.data is None
            )

        for i, choice in enumerate(self.choices.keys()):
            yield Option(
                id='{}_{}'.format(self.name, i),
                value=choice,
                label=str(choice).title(),
                checked=self.data == self.choices.get(choice)
            )

    def fill_from_post(self, post_data):
        super(SimpleSelectField, self).fill_from_post(post_data)
        self.data = self.choices.get(self.data)


class SimpleArrayField(Field):
    template = 'form_fields/str_field.html'

    def __init__(self, form, name, dtype, disabled=False, required=False, label='', doc='', default=None):
        super(SimpleArrayField, self).__init__(form, name, disabled, required, label, doc, default)
        self.dtype = dtype

    def _from_post(self):
        if self.unvalidated_data is not None and isinstance(self.unvalidated_data, str):
            data = json.loads(self.unvalidated_data)
            self.data = numpy.array(data, dtype=self.dtype).tolist()
        elif self.unvalidated_data is not None and isinstance(self.unvalidated_data, list):
            self.data = self.unvalidated_data

    @property
    def value(self):
        if self.data is None:
            # todo: maybe we need to distinguish None from missing data
            # this None means self.data is missing, either not set or unset cause of validation error
            return self.unvalidated_data
        try:
            return json.dumps(self.data.tolist())
        except (TypeError, ValueError):
            return self.unvalidated_data


class DataTypeSelectField(Field):
    template = 'form_fields/datatype_select_field.html'
    missing_value = 'explicit-None-value'

    def __init__(self, datatype_index, form, name=None, disabled=False, required=False, label='', doc='',
                 conditions=None, draw_dynamic_conditions_buttons=True, dynamic_conditions=None, has_all_option=False):
        super(DataTypeSelectField, self).__init__(form, name, disabled, required, label, doc)
        self.datatype_index = datatype_index
        self.conditions = conditions
        self.draw_dynamic_conditions_buttons = draw_dynamic_conditions_buttons
        self.dynamic_conditions = dynamic_conditions
        self.has_all_option = has_all_option

    @property
    def get_dynamic_filters(self):
        return FilterChain().get_filters_for_type(self.datatype_index)

    def _get_values_from_db(self):
        all_conditions = FilterChain()
        all_conditions += self.conditions
        all_conditions += self.dynamic_conditions
        filtered_datatypes, count = dao.get_values_of_datatype(self.owner.project_id,
                                                               self.datatype_index,
                                                               all_conditions)
        return filtered_datatypes

    def options(self):
        if not self.owner.project_id:
            raise ValueError('A project_id is required in order to query the DB')

        filtered_datatypes = self._get_values_from_db()

        if not self.required:
            choice = None
            yield Option(
                id='{}_{}'.format(self.name, None),
                value=self.missing_value,
                label=str(choice).title(),
                checked=self.data is None
            )

        for i, datatype in enumerate(filtered_datatypes):
            yield Option(
                id='{}_{}'.format(self.name, i),
                value=datatype[2],
                label=self._prepare_display_name(datatype),
                checked=self.data == datatype[2]
            )

        if self.has_all_option:
            if not self.owner.draw_ranges:
                raise ValueError("The owner form should draw ranges inputs in order to support 'All' option")

            all_values = ''
            for fdt in filtered_datatypes:
                all_values += str(fdt[2]) + ','

            choice = "All"
            yield Option(
                id='{}_{}'.format(self.name, choice),
                value=all_values[:-1],
                label=choice,
                checked=self.data is choice
            )

    def get_dt_from_db(self):
        return dao.get_datatype_by_gid(self.data)

    def _prepare_display_name(self, value):
        # TODO remove duplicate with TraitedDataTypeSelectField
        """
        Populate meta-data fields for data_list (list of DataTypes).

        Private method, to be called recursively.
        It will receive a list of Attributes, and it will populate 'options'
        entry with data references from DB.
        """
        # Here we only populate with DB data, actual
        # XML check will be done after select and submit.
        entity_gid = value[2]
        actual_entity = dao.get_generic_entity(self.datatype_index, entity_gid, "gid")
        display_name = actual_entity[0].display_name
        display_name += ' - ' + (value[3] or "None ")
        if value[5]:
            display_name += ' - From: ' + str(value[5])
        else:
            display_name += utils.date2string(value[4])
        if value[6]:
            display_name += ' - ' + str(value[6])
        display_name += ' - ID:' + str(value[0])

        return display_name

    def _from_post(self):
        if self.unvalidated_data == self.missing_value:
            self.unvalidated_data = None

        if self.required and not self.unvalidated_data:
            raise ValueError('Field required')

        # TODO: ensure is in choices
        self.data = self.unvalidated_data


class TraitField(Field):
    # mhtodo: while this is consistent with the h5 api, it has the same problem
    #         it couples the system to traited attr declarations
    def __init__(self, trait_attribute, form, name=None, disabled=False):
        # type: (Attr, Form, str, bool) -> None
        self.trait_attribute = trait_attribute  # type: Attr
        name = name or trait_attribute.field_name
        label = trait_attribute.label or name

        super(TraitField, self).__init__(
            form,
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

    def __init__(self, traited_attribute, required_type, form, name, disabled=False):
        super(TraitUploadField, self).__init__(traited_attribute, form, name, disabled)
        self.required_type = required_type
        self.files_helper = FilesHelper()

    def fill_from_post(self, post_data):
        super(TraitUploadField, self).fill_from_post(post_data)

        if self.data.file is None:
            self.data = None
            return

        project = dao.get_project_by_id(self.owner.project_id)
        temporary_storage = self.files_helper.get_project_folder(project, self.files_helper.TEMP_FOLDER)

        file_name = None
        try:
            uq_name = utils.date2string(datetime.now(), True) + '_' + str(0)
            file_name = TEMPORARY_PREFIX + uq_name + '_' + self.data.filename
            file_name = os.path.join(temporary_storage, file_name)

            with open(file_name, 'wb') as file_obj:
                file_obj.write(self.data.file.read())
        except Exception as excep:
            # TODO: is this handled properly?
            self.files_helper.remove_files([file_name])
            excep.message = 'Could not continue: Invalid input files'
            raise excep

        if file_name:
            self.data = file_name
            self.owner.temporary_files.append(file_name)


class TraitDataTypeSelectField(TraitField):
    template = 'form_fields/datatype_select_field.html'
    missing_value = 'explicit-None-value'

    def __init__(self, trait_attribute, form, name=None, conditions=None, draw_dynamic_conditions_buttons=True,
                 dynamic_conditions=None, has_all_option=False):
        super(TraitDataTypeSelectField, self).__init__(trait_attribute, form, name)
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
        self.dynamic_conditions = dynamic_conditions
        self.has_all_option = has_all_option

    def from_trait(self, trait, f_name):
        if hasattr(trait, f_name):
            self.data = getattr(trait, f_name)

    @property
    def get_dynamic_filters(self):
        return FilterChain().get_filters_for_type(self.datatype_index)

    def _get_values_from_db(self):
        all_conditions = FilterChain()
        all_conditions += self.conditions
        all_conditions += self.dynamic_conditions
        filtered_datatypes, count = dao.get_values_of_datatype(self.owner.project_id,
                                                               self.datatype_index,
                                                               all_conditions)
        return filtered_datatypes

    def options(self):
        if not self.owner.project_id:
            raise ValueError('A project_id is required in order to query the DB')

        filtered_datatypes = self._get_values_from_db()

        if not self.required:
            choice = None
            yield Option(
                id='{}_{}'.format(self.name, None),
                value=self.missing_value,
                label=str(choice).title(),
                checked=self.data is None
            )

        for i, datatype in enumerate(filtered_datatypes):
            yield Option(
                id='{}_{}'.format(self.name, i),
                value=datatype[2],
                label=self._prepare_display_name(datatype),
                checked=self.data == datatype[2]
            )

        if self.has_all_option:
            if not self.owner.draw_ranges:
                raise ValueError("The owner form should draw ranges inputs in order to support 'All' option")

            all_values = ''
            for fdt in filtered_datatypes:
                all_values += str(fdt[2]) + ','

            choice = "All"
            yield Option(
                id='{}_{}'.format(self.name, choice),
                value=all_values[:-1],
                label=choice,
                checked=self.data is choice
            )

    def get_dt_from_db(self):
        return dao.get_datatype_by_gid(self.data)

    def _prepare_display_name(self, value):
        """
        Populate meta-data fields for data_list (list of DataTypes).

        Private method, to be called recursively.
        It will receive a list of Attributes, and it will populate 'options'
        entry with data references from DB.
        """
        # Here we only populate with DB data, actual
        # XML check will be done after select and submit.
        entity_gid = value[2]
        actual_entity = dao.get_generic_entity(self.datatype_index, entity_gid, "gid")
        display_name = actual_entity[0].display_name
        display_name += ' - ' + (value[3] or "None ")   # Subject
        if value[5]:
            display_name += ' - From: ' + str(value[5])
        else:
            display_name += utils.date2string(value[4])
        if value[6]:
            display_name += ' - ' + str(value[6])
        display_name += ' - ID:' + str(value[0])

        return display_name

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

    def _from_post(self):
        if self.required and (self.unvalidated_data is None or self.unvalidated_data.strip() == ''):
            raise ValueError('Field required')
        self.data = self.unvalidated_data


class BytesField(StrField):
    """ StrField for byte strings. """
    template = 'form_fields/str_field.html'

    def _from_post(self):
        super(BytesField, self)._from_post()
        self.data = self.unvalidated_data.encode('utf-8')


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
        self.data = int(self.unvalidated_data)


class FloatField(TraitField):
    template = 'form_fields/number_field.html'
    input_type = "number"
    min = None
    max = None
    step = 'any'

    def _from_post(self):
        super(FloatField, self)._from_post()
        # TODO: Throws exception if attr is optional and has no value
        if self.unvalidated_data and len(self.unvalidated_data) == 0:
            self.unvalidated_data = None
        if self.unvalidated_data:
            self.data = float(self.unvalidated_data)
        else:
            self.data = None


class ArrayField(TraitField):
    template = 'form_fields/str_field.html'

    def _from_post(self):
        data = json.loads(self.unvalidated_data)
        self.data = numpy.array(data, dtype=self.trait_attribute.dtype)

    @property
    def value(self):
        if self.data is None:
            # todo: maybe we need to distinguish None from missing data
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

    def __init__(self, trait_attribute, form, name=None, disabled=False, choices=None, display_none_choice=True):
        super(SelectField, self).__init__(trait_attribute, form, name, disabled)
        if choices:
            self.choices = choices
        else:
            self.choices = {choice: choice for choice in trait_attribute.choices}
        if not self.choices:
            raise ValueError('no choices for field')
        self.display_none_choice = display_none_choice

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

    # def _from_post(self):
    #     # encode None as a string
    #     if self.unvalidated_data == self.missing_value:
    #         self.unvalidated_data = None
    #
    #     if self.required and not self.unvalidated_data:
    #         raise ValueError('Field required')
    #
    #     if self.unvalidated_data is not None:
    #         # todo muliple values
    #         self.data = self.trait_attribute.field_type(self.unvalidated_data)
    #     else:
    #         self.data = None
    #
    #     allowed = self.trait_attribute.choices
    #     if not self.trait_attribute.required:
    #         allowed = (None,) + allowed
    #
    #     if self.data not in allowed:
    #         raise ValueError('must be one of {}'.format(allowed))

    def fill_from_post(self, post_data):
        super(SelectField, self).fill_from_post(post_data)
        self.data = self.choices.get(self.data)


class MultiSelectField(TraitField):
    template = 'form_fields/checkbox_field.html'

    def __init__(self, trait_attribute, form, name=None, disabled=False):
        super(MultiSelectField, self).__init__(trait_attribute, form, name, disabled)
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


# noinspection PyPep8Naming
def ScalarField(trait_attribute, form, name=None, disabled=False):
    # as this makes introspective decisions it has to be moved at a different level
    field_type_for_trait_type = {
        # str: BytesField,
        str: StrField,
        int: IntField,
        float: FloatField,
        bool: BoolField,
    }

    if trait_attribute.choices is not None:
        cls = SelectField
    else:
        cls = field_type_for_trait_type.get(trait_attribute.field_type)

    if isinstance(trait_attribute, List) and trait_attribute.element_choices:
        cls = MultiSelectField

    if cls is None:
        raise ValueError('can not make a scalar field for trait attribute {}'.format(trait_attribute))

    return cls(trait_attribute, form, name=name, disabled=disabled)


class FormField(Field):
    template = 'form_fields/form_field.html'

    def __init__(self, form_class, form, name, label='', doc=''):
        super(FormField, self).__init__(form, name, False, False, label, doc)
        self.form = form_class(prefix=name)

    def fill_from_post(self, post_data):
        self.form.fill_from_post(post_data)
        self.errors = self.form.errors

    def validate(self):
        return self.form.validate()

    def __str__(self):
        return jinja_env.get_template(self.template).render(adapter_form=self.form)


class Form(object):
    RANGE_1_NAME = 'range_1'
    RANGE_2_NAME = 'range_2'
    range_1 = None
    range_2 = None

    def __init__(self, prefix='', project_id=None, draw_ranges=True):
        # TODO: makes sense here?
        self.project_id = project_id
        self.prefix = prefix
        self.errors = []
        self.draw_ranges = draw_ranges

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

    def fill_from_post(self, form_data):
        for field in self.fields:
            field.fill_from_post(form_data)
        self.range_1 = form_data.get(self.RANGE_1_NAME)
        self.range_2 = form_data.get(self.RANGE_2_NAME)
