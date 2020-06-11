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

from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.neotraits.forms import TraitDataTypeSelectField, DataTypeSelectField


def _review_operation_inputs_for_adapter_model(form_fields, form_model, view_model):
    changed_attr = {}
    inputs_datatypes = []
    for field in form_fields:
        if not isinstance(field, TraitDataTypeSelectField) and not isinstance(field, DataTypeSelectField):
            attr_default = getattr(form_model, field.name)
            attr_vm = getattr(view_model, field.name)
            if attr_vm != attr_default:
                if isinstance(attr_default, float) or isinstance(attr_default, str):
                    changed_attr[field.label] = attr_vm
                else:
                    changed_attr[field.label] = attr_vm.title
        else:
            attr_vm = getattr(view_model, field.name)
            data_type = ABCAdapter.load_entity_by_gid(attr_vm)
            if attr_vm:
                changed_attr[field.label] = data_type.display_name
            inputs_datatypes.append(data_type)

    return inputs_datatypes, changed_attr


def review_operation_inputs_from_adapter(adapter, operation):
    """
    :returns: a list with the inputs from the parameters list that are instances of DataType,\
        and a dictionary with all parameters which are different than the declared defauts
    """
    view_model = adapter.load_view_model(operation)
    form_model = adapter.get_view_model_class()()
    form_fields = adapter.get_form_class()().fields

    if 'SimulatorAdapter' in operation.algorithm.classname:
        fragments = adapter.get_simulator_fragments()
        inputs_datatypes, changed_attr = _review_operation_inputs_for_adapter_model(form_fields, form_model, view_model)
        for fragment in fragments:
            fragment_fields = fragment().fields
            for field in fragment_fields:
                if hasattr(view_model, field.name):
                    if not isinstance(field, TraitDataTypeSelectField) and not isinstance(field, DataTypeSelectField):
                        attr_default = getattr(form_model, field.name)
                        attr_vm = getattr(view_model, field.name)
                        if attr_vm != attr_default:
                            if isinstance(attr_default, float) or isinstance(attr_default, str):
                                changed_attr[field.label] = attr_vm
                            else:
                                if not isinstance(attr_default, tuple):
                                    changed_attr[field.label] = attr_vm.title
                                else:
                                    for sub_attr in attr_default:
                                        changed_attr[field.label] = sub_attr.title
                    else:
                        attr_vm = getattr(view_model, field.name)
                        data_type = ABCAdapter.load_entity_by_gid(attr_vm)
                        if attr_vm:
                            changed_attr[field.label] = data_type.display_name
                        inputs_datatypes.append(data_type)
    else:
        inputs_datatypes, changed_attr = _review_operation_inputs_for_adapter_model(form_fields, form_model, view_model)

    return inputs_datatypes, changed_attr
