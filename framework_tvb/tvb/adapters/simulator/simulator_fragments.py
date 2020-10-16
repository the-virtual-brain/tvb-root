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

import uuid

import formencode
from formencode import validators
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex, SpatioTemporalPatternIndex
from tvb.adapters.simulator.integrator_forms import get_form_for_integrator
from tvb.adapters.simulator.model_forms import get_ui_name_to_model
from tvb.adapters.simulator.monitor_forms import get_ui_name_to_monitor_dict, get_monitor_to_ui_name_dict
from tvb.adapters.simulator.subforms_mapping import get_ui_name_to_integrator_dict
from tvb.basic.neotraits.api import Attr, Range, List
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.entities.file.simulator.view_model import CortexViewModel, SimulatorAdapterModel, IntegratorViewModel
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import ScalarField, ArrayField, SimpleFloatField, SimpleHiddenField, SelectField, \
    MultiSelectField, TraitDataTypeSelectField
from tvb.core.neotraits.view_model import Str
from tvb.datatypes.surfaces import CORTICAL
from tvb.simulator.models.base import Model


class SimulatorSurfaceFragment(ABCAdapterForm):
    def __init__(self, prefix='', project_id=None):
        super(SimulatorSurfaceFragment, self).__init__(prefix, project_id)
        conditions = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=["=="],
                                 values=[CORTICAL])
        self.surface = TraitDataTypeSelectField(CortexViewModel.surface_gid, self, name='surface',
                                                conditions=conditions)

    def fill_trait(self, datatype):
        surface_gid = self.surface.value
        if surface_gid:
            if not datatype.surface or (datatype.surface and datatype.surface.surface_gid != surface_gid):
                datatype.surface = CortexViewModel()
                datatype.surface.surface_gid = surface_gid
        else:
            datatype.surface = None


class SimulatorRMFragment(ABCAdapterForm):
    def __init__(self, prefix='', project_id=None, surface_index=None, connectivity_gid=None):
        super(SimulatorRMFragment, self).__init__(prefix, project_id)
        rm_conditions = None
        lc_conditions = None
        if surface_index:
            rm_conditions = FilterChain(fields=[FilterChain.datatype + '.fk_surface_gid',
                                             FilterChain.datatype + '.fk_connectivity_gid'],
                                     operations=["==", "=="],
                                     values=[str(surface_index.gid), str(connectivity_gid.hex)])
            lc_conditions = FilterChain(fields=[rm_conditions.fields[0]], operations=[rm_conditions.operations[0]],
                                        values=[rm_conditions.values[0]])
        self.rm = TraitDataTypeSelectField(CortexViewModel.region_mapping_data, self, name='region_mapping',
                                           conditions=rm_conditions)

        self.lc = TraitDataTypeSelectField(CortexViewModel.local_connectivity, self, name='local_connectivity',
                                           conditions=lc_conditions)
        self.coupling_strength = ArrayField(CortexViewModel.coupling_strength, self)


class SimulatorStimulusFragment(ABCAdapterForm):
    def __init__(self, prefix='', project_id=None, is_surface_simulation=False):
        super(SimulatorStimulusFragment, self).__init__(prefix, project_id)
        stimuli_index_class = StimuliRegionIndex
        if is_surface_simulation:
            stimuli_index_class = SpatioTemporalPatternIndex
        traited_field = Attr(stimuli_index_class, doc=SimulatorAdapterModel.stimulus.doc,
                             label=SimulatorAdapterModel.stimulus.label,
                             required=SimulatorAdapterModel.stimulus.required)
        self.stimulus = TraitDataTypeSelectField(traited_field, self, name='stimulus')

    def fill_trait(self, datatype):
        setattr(datatype, self.stimulus.name, self.stimulus.data)

    def fill_from_trait(self, trait):
        self.stimulus.from_trait(trait, self.stimulus.name)


class SimulatorModelFragment(ABCAdapterForm):
    def __init__(self, prefix='', project_id=None):
        super(SimulatorModelFragment, self).__init__(prefix, project_id)
        self.model_choices = get_ui_name_to_model()
        default_model = list(self.model_choices.values())[0]

        self.model = SelectField(
            Attr(Model, default=default_model, label=SimulatorAdapterModel.model.label,
                 doc=SimulatorAdapterModel.model.doc), self,
            choices=self.model_choices, name='model')

    def fill_from_trait(self, trait):
        # type: (SimulatorAdapterModel) -> None
        self.model.data = trait.model.__class__

    def fill_trait(self, datatype):
        if type(datatype.model) != self.model.value:
            datatype.model = self.model.value()


class SimulatorIntegratorFragment(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(SimulatorIntegratorFragment, self).__init__(prefix, project_id)
        self.integrator_choices = get_ui_name_to_integrator_dict()
        default_integrator = list(self.integrator_choices.values())[0]

        self.integrator = SelectField(
            Attr(IntegratorViewModel, default=default_integrator, label=SimulatorAdapterModel.integrator.label,
                 doc=SimulatorAdapterModel.integrator.doc), self, name='integrator',
            choices=self.integrator_choices,
            subform=get_form_for_integrator(default_integrator))

    def fill_from_trait(self, trait):
        # type: (SimulatorAdapterModel) -> None
        self.integrator.data = trait.integrator.__class__

    def fill_trait(self, datatype):
        if type(datatype.integrator) != self.integrator.value:
            datatype.integrator = self.integrator.value()


class SimulatorMonitorFragment(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None, is_surface_simulation=False):
        super(SimulatorMonitorFragment, self).__init__(prefix, project_id)
        self.monitor_choices = get_ui_name_to_monitor_dict(is_surface_simulation)
        self.is_surface_simulation = is_surface_simulation

        self.monitors = MultiSelectField(List(of=str, label='Monitors',
                                              choices=tuple(self.monitor_choices.keys())),
                                         self, name='monitors')

    def fill_from_trait(self, trait):
        # type: (SimulatorAdapterModel) -> None
        self.monitors.data = [
            get_monitor_to_ui_name_dict(self.is_surface_simulation)[type(monitor)]
            for monitor in trait]


class SimulatorFinalFragment(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None, default_simulation_name="simulation_1"):
        super(SimulatorFinalFragment, self).__init__(prefix, project_id)
        self.simulation_length = ScalarField(SimulatorAdapterModel.simulation_length, self)
        self.simulation_name = ScalarField(Attr(str, doc='Name for the current simulation configuration',
                                                default=default_simulation_name, label='Simulation name'), self,
                                           name='input_simulation_name_id')

    def fill_from_post(self, form_data):
        super(SimulatorFinalFragment, self).fill_from_post(form_data)
        valiadation_result = SimulatorFinalFragment.is_burst_name_ok(self.simulation_name.value)
        if valiadation_result is not True:
            raise ValueError(valiadation_result)

    @staticmethod
    def is_burst_name_ok(burst_name):
        class BurstNameForm(formencode.Schema):
            """
            Validate Burst name string
            """
            burst_name = formencode.All(validators.UnicodeString(not_empty=True),
                                        validators.Regex(regex=r"^[a-zA-Z\. _\-0-9]*$"))

        try:
            form = BurstNameForm()
            form.to_python({'burst_name': burst_name})
            return True
        except formencode.Invalid:
            validation_error = "Invalid simulation name %s. Please use only letters, numbers, or _ " % str(burst_name)
            return validation_error


class SimulatorPSEConfigurationFragment(ABCAdapterForm):

    def __init__(self, choices, prefix='', project_id=None):
        super(SimulatorPSEConfigurationFragment, self).__init__(prefix, project_id)
        default_choice = list(choices.values())[0]
        self.pse_param1 = SelectField(Str(default=default_choice, label="PSE param1"), self, choices=choices,
                                      name='pse_param1')
        self.pse_param2 = SelectField(Str(label="PSE param2", required=False), self, choices=choices, name='pse_param2')


class SimulatorPSERangeFragment(ABCAdapterForm):
    KEY_PARAM1 = 'param1'
    KEY_PARAM2 = 'param2'
    NAME_FIELD = 'pse_{}_name'
    LO_FIELD = 'pse_{}_lo'
    HI_FIELD = 'pse_{}_hi'
    STEP_FIELD = 'pse_{}_step'
    GID_FIELD = 'pse_{}_guid'

    def __init__(self, pse_param1, pse_param2, prefix='', project_id=None):
        # type: (RangeParameter, RangeParameter, str, int) -> None
        super(SimulatorPSERangeFragment, self).__init__(prefix, project_id)
        self._add_pse_field(pse_param1)
        if pse_param2:
            self._add_pse_field(pse_param2, self.KEY_PARAM2)

    def _add_pse_field(self, param, param_key=KEY_PARAM1):
        # type: (RangeParameter, str) -> None
        pse_param_name = SimpleHiddenField(self, name=self.NAME_FIELD.format(param_key), default=param.name)
        self.__setattr__(self.NAME_FIELD.format(param_key), pse_param_name)
        if param.type is float:
            self._add_fields_for_float(param, param_key)
        else:
            self._add_field_for_gid(param, param_key)

    def _add_fields_for_float(self, param, param_key):
        # type: (RangeParameter, str) -> None
        pse_param_lo = SimpleFloatField(self, name=self.LO_FIELD.format(param_key), required=True,
                                        label='LO for {}'.format(param.name), default=param.range_definition.lo)
        self.__setattr__(self.LO_FIELD.format(param_key), pse_param_lo)
        pse_param_hi = SimpleFloatField(self, name=self.HI_FIELD.format(param_key), required=True,
                                        label='HI for {}'.format(param.name), default=param.range_definition.hi)
        self.__setattr__(self.HI_FIELD.format(param_key), pse_param_hi)
        pse_param_step = SimpleFloatField(self, name=self.STEP_FIELD.format(param_key), required=True,
                                          label='STEP for {}'.format(param.name), default=param.range_definition.step)
        self.__setattr__(self.STEP_FIELD.format(param_key), pse_param_step)

    def _add_field_for_gid(self, param, param_key):
        # type: (RangeParameter, str) -> None
        traited_attr = Attr(h5.REGISTRY.get_index_for_datatype(param.type), label='Choice for {}'.format(param.name))
        pse_param_dt = TraitDataTypeSelectField(traited_attr, self, name=self.GID_FIELD.format(param_key),
                                                dynamic_conditions=param.range_definition, has_all_option=True,
                                                show_only_all_option=True)
        self.__setattr__(self.GID_FIELD.format(param_key), pse_param_dt)

    @staticmethod
    def _prepare_pse_uuid_list(pse_uuid_str):
        pse_uuid_str_list = pse_uuid_str.split(',')
        pse_uuid_list = [uuid.UUID(uuid_str) for uuid_str in pse_uuid_str_list]
        return pse_uuid_list

    @staticmethod
    def _fill_param_from_post(all_range_parameters, param_key, **data):
        # type: (dict, str, dict) -> RangeParameter
        param_name = data.get(SimulatorPSERangeFragment.NAME_FIELD.format(param_key))
        param = all_range_parameters.get(param_name)
        if param.type is float:
            param_lo = data.get(SimulatorPSERangeFragment.LO_FIELD.format(param_key))
            param_hi = data.get(SimulatorPSERangeFragment.HI_FIELD.format(param_key))
            param_step = data.get(SimulatorPSERangeFragment.STEP_FIELD.format(param_key))
            param_range = RangeParameter(param_name, param.type,
                                         Range(float(param_lo), float(param_hi), float(param_step)),
                                         is_array=param.is_array)
        else:
            param_range_str = data.get(SimulatorPSERangeFragment.GID_FIELD.format(param_key))
            param_range = RangeParameter(param_name, param.type, param.range_definition, False,
                                         SimulatorPSERangeFragment._prepare_pse_uuid_list(param_range_str))
        return param_range

    @staticmethod
    def fill_from_post(all_range_parameters, **data):
        # type: (dict, dict) -> tuple
        param1 = SimulatorPSERangeFragment._fill_param_from_post(all_range_parameters,
                                                                 SimulatorPSERangeFragment.KEY_PARAM1, **data)
        pse_param2_name = data.get(SimulatorPSERangeFragment.NAME_FIELD.format(SimulatorPSERangeFragment.KEY_PARAM2))

        param2 = None
        if pse_param2_name:
            param2 = SimulatorPSERangeFragment._fill_param_from_post(all_range_parameters,
                                                                     SimulatorPSERangeFragment.KEY_PARAM2, **data)

        return param1, param2
