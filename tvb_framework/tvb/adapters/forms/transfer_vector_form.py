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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""
from tvb.adapters.forms.equation_forms import TransferVectorEquationsEnum, get_form_for_equation
from tvb.basic.neotraits._attr import EnumAttr
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.neotraits.forms import DynamicSelectField, TraitDataTypeSelectField, SelectField
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.datatypes.graph import ConnectivityMeasure

KEY_TRANSFER = "transferVector"


class TransferVectorForm(ABCAdapterForm):
    default_transfer_function = TransferVectorEquationsEnum.IDENTITY
    transfer_function_label = 'Transfer Function'

    def __init__(self, model_params):
        super(TransferVectorForm, self).__init__()

        model_labels = [param.name for param in model_params]
        model_mathjax_representations = [param.label for param in model_params]
        self.model_param = DynamicSelectField(Str(label='Model parameter'), choices=model_labels, name='model_param',
                                              ui_values=model_mathjax_representations)

        cm_attribute = DataTypeGidAttr(
            linked_datatype=ConnectivityMeasure,
            label='Original Spatial Vector',
            doc='A previously stored compatible Spatial Vector'
        )
        cm_filter = FilterChain(fields=[FilterChain.datatype + '.ndim'], operations=['=='], values=['1'])
        self.connectivity_measure = TraitDataTypeSelectField(cm_attribute, name='connectivity_measure',
                                                             conditions=cm_filter)

        transfer_function_attribute = EnumAttr(
            field_type=TransferVectorEquationsEnum,
            label=self.transfer_function_label,
            default=self.default_transfer_function
        )
        self.transfer_function = SelectField(transfer_function_attribute, name='transfer_function',
                                             subform=get_form_for_equation(self.default_transfer_function.value),
                                             session_key=KEY_TRANSFER)
