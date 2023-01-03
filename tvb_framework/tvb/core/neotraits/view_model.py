# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

import uuid
import numpy
from datetime import datetime
from tvb.basic.neotraits.api import HasTraits, Attr
from tvb.basic.neotraits.ex import TraitAttributeError
from tvb.core.entities.generic_attributes import GenericAttributes


class ViewModel(HasTraits):
    operation_group_gid = Attr(field_type=uuid.UUID, required=False)
    ranges = Attr(str, required=False)
    range_values = Attr(str, required=False)
    is_metric_operation = Attr(bool, default=False)

    def __init__(self, **kwargs):
        super(ViewModel, self).__init__(**kwargs)
        self.create_date = datetime.now()
        self.generic_attributes = GenericAttributes()

    def linked_has_traits(self):
        return HasTraits


class Str(Attr):
    def __init__(self, field_type=str, default=None, doc='', label='', required=True,
                 final=False, choices=None):
        super(Str, self).__init__(field_type, default, doc, label, required, final, choices)


class DataTypeGidAttr(Attr):
    """
    Keep a GID but also link the type of DataType it should point to
    """

    def __init__(self, linked_datatype, field_type=uuid.UUID, filters=None, default=None, doc='', label='',
                 required=True, final=False, choices=None):
        super(DataTypeGidAttr, self).__init__(field_type, default, doc, label, required, final, choices)
        self.linked_datatype = linked_datatype
        self.filters = filters

    def __set__(self, instance, value):
        if isinstance(value, str):
            try:
                value = uuid.UUID(value)
            except ValueError:
                raise TraitAttributeError("Given value cannot be used as UUID for field {}".format(self.field_name))

        super(DataTypeGidAttr, self).__set__(instance, value)


class EquationAttr(Attr):
    """
    # TODO: there are places where we need eq params as a nested form. Figure out a proper model
    """


def replace_nan_values(input_data):
    """ Replace NAN values with a given values"""
    is_any_value_nan = False
    if not numpy.isfinite(input_data).all():
        for idx in range(len(input_data)):
            input_data[idx] = numpy.nan_to_num(input_data[idx])
        is_any_value_nan = True
    return is_any_value_nan
