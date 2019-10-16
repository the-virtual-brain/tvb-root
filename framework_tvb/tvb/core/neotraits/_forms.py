import json
from collections import namedtuple

import numpy
from tvb.core import utils
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.storage import dao
from tvb.basic.neotraits.ex import TraitError
from tvb.basic.neotraits.api import List, Attr

# This setting is injected.
# The pattern might be confusing, but it is an interesting alternative to
# universal tvbprofile imports

jinja_env = None


class Field(object):
    template = None

    def __init__(self, form, name, disabled=False, required=False, label='', doc=''):
        # type: (Form, str, bool, bool, str, str) -> None
        self.owner = form
        self.name = '{}_{}'.format(form.prefix, name)
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
        self.unvalidated_data = None
        self.errors = []


    def fill_from_post(self, post_data):
        """ deserialize form a post dictionary """
        self.unvalidated_data = post_data.get(self.name)
        try:
            self._from_post()
        except ValueError as ex:
            self.errors.append(ex.message)

    def validate(self):
        """ validation besides the deserialization from post"""
        return not self.errors

    def _from_post(self):
        if self.required and self.unvalidated_data is None:
            raise ValueError('Field required')
        self.data = self.unvalidated_data

    @property
    def value(self):
        return self.data or self.unvalidated_data

    def __repr__(self):
        return '<{}>(name={})'.format(type(self).__name__, self.name)

    def __str__(self):
        return jinja_env.get_template(self.template).render(field=self)


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
            trait_attribute.doc
        )

class DataTypeSelectField(TraitField):
    template = 'datatype_select_field.jinja2'
    missing_value = 'explicit-None-value'

    def __init__(self, trait_attribute, datatype_index, form, name=None, disabled=False):
        super(DataTypeSelectField, self).__init__(trait_attribute, form, name, disabled)
        self.datatype_index = datatype_index

    def _get_values_from_db(self):
        filtered_datatypes, count = dao.get_values_of_datatype(self.owner.project_id,
                                                               self.datatype_index,
                                                               self.owner.get_filters())
        return filtered_datatypes

    def options(self):
        if not self.owner.project_id:
            raise ValueError('A project_id is required in order to query the DB')

        filtered_datatypes = self._get_values_from_db()

        if not self.trait_attribute.required:
            choice = None
            yield Option(
                id='{}_{}'.format(self.name, None),
                value=self.missing_value,
                label=str(choice).title(),
                checked=self.data is None
            )

        # TODO: Add "All" option
        for i, datatype in enumerate(filtered_datatypes):
            yield Option(
                id='{}_{}'.format(self.name, i),
                value=datatype[2],
                label=self._prepare_display_name(datatype),
                checked=self.data == datatype
            )

    def get_dt_from_db(self):
        return dao.get_time_series_by_gid(self.data)

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
        display_name = ''
        if actual_entity is not None and len(actual_entity) > 0 and isinstance(actual_entity[0], DataType):
            display_name = actual_entity[0].__class__.__name__
        display_name += ' - ' + (value[3] or "None ")
        if value[5]:
            display_name += ' - From: ' + str(value[5])
        else:
            display_name += utils.date2string(value[4])
        if value[6]:
            display_name += ' - ' + str(value[6])
        display_name += ' - ID:' + str(value[0])

        return display_name


class TimeSeriesSelectField(DataTypeSelectField):
    def _get_values_from_db(self):
        filtered_ts, count = dao.get_values_of_time_series(self.owner.project_id,
                                                           self.datatype_index,
                                                           self.owner.get_filters())
        return filtered_ts

class StrField(TraitField):
    template = 'str_field.jinja2'

    def _from_post(self):
        if self.required and (self.unvalidated_data is None or self.unvalidated_data.strip() == ''):
            raise ValueError('Field required')
        self.data = self.unvalidated_data


class BytesField(StrField):
    """ StrField for byte strings. """
    template = 'str_field.jinja2'

    def _from_post(self):
        super(BytesField, self)._from_post()
        self.data = self.unvalidated_data.encode('utf-8')


class BoolField(TraitField):
    template = 'bool_field.jinja2'

    def _from_post(self):
        self.data = self.unvalidated_data is not None


class IntField(TraitField):
    template = 'number_field.jinja2'
    min = None
    max = None

    def _from_post(self):
        super(IntField, self)._from_post()
        self.data = int(self.unvalidated_data)


class FloatField(TraitField):
    template = 'number_field.jinja2'
    input_type = "number"
    min = None
    max = None
    step = 'any'

    def _from_post(self):
        super(FloatField, self)._from_post()
        self.data = float(self.unvalidated_data)


class ArrayField(TraitField):
    template = 'str_field.jinja2'

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
            return json.dumps(self.data.tolist())
        except (TypeError, ValueError):
            return self.unvalidated_data


Option = namedtuple('Option', ['id', 'value', 'label', 'checked'])


class SelectField(TraitField):
    template = 'radio_field.jinja2'
    missing_value = 'explicit-None-value'

    def __init__(self, trait_attribute, form, name=None, disabled=False):
        super(SelectField, self).__init__(trait_attribute, form, name, disabled)
        if not trait_attribute.choices:
            raise ValueError('no choices for field')

    def options(self):
        """ to be used from template, assumes self.data is set """
        if not self.trait_attribute.required:
            choice = None
            yield Option(
                id='{}_{}'.format(self.name, None),
                value=self.missing_value,
                label=str(choice).title(),
                checked=self.data is None
            )

        for i, choice in enumerate(self.trait_attribute.choices):
            yield Option(
                id='{}_{}'.format(self.name, i),
                value=choice,
                label=str(choice).title(),
                checked=self.data == choice
            )

    def _from_post(self):
        # encode None as a string
        if self.unvalidated_data == self.missing_value:
            self.unvalidated_data = None

        if self.required and not self.unvalidated_data:
            raise ValueError('Field required')

        if self.unvalidated_data is not None:
            # todo muliple values
            self.data = self.trait_attribute.field_type(self.unvalidated_data)
        else:
            self.data = None

        allowed = self.trait_attribute.choices
        if not self.trait_attribute.required:
            allowed = (None, ) + allowed

        if self.data not in allowed:
           raise ValueError('must be one of {}'.format(allowed))



class MultiSelectField(TraitField):
    template = 'checkbox_field.jinja2'

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
                checked=choice in self.data
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

        data = []   # don't mutate self.data until we know all values converted ok

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
        str: BytesField,
        unicode: StrField,
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
    template = 'form_field.jinja2'

    def __init__(self, form_class, form, name, label='', doc=''):
        super(FormField, self).__init__(form, name, False, False, label, doc)
        self.form = form_class(prefix=name)

    def fill_from_post(self, post_data):
        self.form.fill_from_post(post_data)
        self.errors = self.form.errors

    def validate(self):
        return self.form.validate()

    def __str__(self):
        return jinja_env.get_template(self.template).render(form=self.form)


class Form(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.errors = []

    @property
    def fields(self):
        for field in self.__dict__.itervalues():
            if isinstance(field, Field):
                yield field

    @property
    def trait_fields(self):
        for field in self.__dict__.itervalues():
            if isinstance(field, TraitField):
                if isinstance(field, DataTypeSelectField):
                    continue
                yield field

    def validate(self):
        valid = True
        for field in self.fields:
            if not field.validate():
                valid = False
        return valid

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
            field.data = getattr(trait, f_name)


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
                field.errors.append(ex.message)
                raise

    def fill_from_post(self, form_data):
        for field in self.fields:
            field.fill_from_post(form_data)
