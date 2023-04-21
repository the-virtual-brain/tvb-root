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

"""
.. moduleauthor:: Adrian Dordea <adrian.dordea@codemart.ro>
"""

import os
import numpy
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.neotraits.forms import TraitDataTypeSelectField, TraitUploadField, UserSessionStrField


def _review_operation_inputs_for_adapter_model(form_fields, form_model, view_model):
    changed_attr = {}
    inputs_datatypes = []

    for field in form_fields:

        if not hasattr(view_model, field.name):
            continue
        attr_vm = getattr(view_model, field.name)
        if attr_vm and type(field) == TraitUploadField:
            attr_vm = os.path.basename(attr_vm)
        if attr_vm and type(field) == UserSessionStrField:
            # Don't show UserSession actual value, as these might contain a secret, instead show the env Variable
            changed_attr[field.label] = "SECRET ${%s}" % field.name
            continue

        if isinstance(field, TraitDataTypeSelectField):
            data_type = None
            if attr_vm:
                data_type = load_entity_by_gid(attr_vm)
                changed_attr[field.label] = data_type.display_name if data_type else "None"
            inputs_datatypes.append(data_type)
        else:
            attr_default = None
            if hasattr(form_model, field.name):
                attr_default = getattr(form_model, field.name)

            if isinstance(attr_vm, numpy.ndarray):
                check_for_changed = attr_vm.size != 0
            else:
                check_for_changed = attr_vm != attr_default

            if check_for_changed:
                if isinstance(attr_vm, float) or isinstance(attr_vm, int) or isinstance(attr_vm, str):
                    changed_attr[field.label] = attr_vm
                elif isinstance(attr_vm, tuple) or isinstance(attr_vm, list):
                    changed_attr[field.label] = ', '.join([str(sub_attr) for sub_attr in attr_vm])
                else:
                    # All HasTraits instances will show as being different than default, even if the same!! Is this ok?
                    changed_attr[field.label] = str(attr_vm)

    return inputs_datatypes, changed_attr


def review_operation_inputs_from_adapter(adapter, operation):
    """
    :returns: a list with the inputs from the parameters list that are instances of DataType,\
        and a dictionary with all parameters which are different than the declared defauts
    """
    view_model = adapter.load_view_model(operation)
    form_model = adapter.get_view_model_class()()
    form_fields = adapter.get_form_class()().fields

    inputs_datatypes, changed_attr = _review_operation_inputs_for_adapter_model(form_fields, form_model, view_model)

    fragments_dict = adapter.get_adapter_fragments(view_model)
    # The Simulator, for example will have Fragments
    for path, fragments in fragments_dict.items():
        if path is None:
            fragment_defaults = form_model
            fragment_model = view_model
        else:
            fragment_defaults = getattr(form_model, path)
            fragment_model = getattr(view_model, path)

        for fragment in fragments:
            fragment_fields = fragment().fields

            part_dts, part_changed = _review_operation_inputs_for_adapter_model(fragment_fields,
                                                                                fragment_defaults,
                                                                                fragment_model)
            inputs_datatypes.extend(part_dts)
            changed_attr.update(part_changed)

    return inputs_datatypes, changed_attr
