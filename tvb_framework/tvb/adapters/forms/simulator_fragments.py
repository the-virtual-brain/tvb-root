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
import formencode
from formencode import validators

from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex, SpatioTemporalPatternIndex
from tvb.adapters.forms.form_with_ranges import FormWithRanges
from tvb.adapters.forms.integrator_forms import get_form_for_integrator, get_integrator_name_list
from tvb.adapters.forms.model_forms import ModelsEnum
from tvb.adapters.forms.monitor_forms import get_ui_name_to_monitor_dict, get_monitor_to_ui_name_dict
from tvb.basic.profile import TvbProfile
from tvb.basic.neotraits.api import Attr, EnumAttr, Range, List, Float, Int
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.entities.file.simulator.view_model import CortexViewModel, SimulatorAdapterModel, IntegratorViewModelsEnum
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.entities.transient.range_parameter import RangeParameter
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import ArrayField, SelectField, MultiSelectField, \
    TraitDataTypeSelectField, HiddenField, FloatField, StrField, DynamicSelectField
from tvb.core.neotraits.view_model import Str
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.burst_service import BurstService
from tvb.datatypes.surfaces import SurfaceTypesEnum


class SimulatorSurfaceFragment(ABCAdapterForm):
    def __init__(self):
        super(SimulatorSurfaceFragment, self).__init__()
        conditions = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=["=="],
                                 values=[SurfaceTypesEnum.CORTICAL_SURFACE.value])
        self.surface = TraitDataTypeSelectField(CortexViewModel.surface_gid, name='surface_gid', conditions=conditions)

    def fill_trait(self, datatype):
        surface_gid = self.surface.value
        if surface_gid:
            if not datatype.surface or (datatype.surface and datatype.surface.surface_gid != surface_gid):
                datatype.surface = CortexViewModel()
                datatype.surface.surface_gid = surface_gid
        else:
            datatype.surface = None

    @staticmethod
    def prepare_next_fragment_after_surface(simulator, rendering_rules, project_id, cortex_url, stimulus_url):
        if simulator.surface:
            return SimulatorRMFragment.prepare_cortex_fragment(simulator, rendering_rules, cortex_url, project_id)
        return SimulatorStimulusFragment.prepare_stimulus_fragment(simulator, rendering_rules, False, stimulus_url,
                                                                   project_id)


class SimulatorRMFragment(FormWithRanges):
    def __init__(self, surface_index=None, connectivity_gid=None):
        super(SimulatorRMFragment, self).__init__()
        rm_conditions = None
        lc_conditions = None
        if surface_index:
            rm_conditions = FilterChain(fields=[FilterChain.datatype + '.fk_surface_gid',
                                                FilterChain.datatype + '.fk_connectivity_gid'],
                                        operations=["==", "=="],
                                        values=[str(surface_index.gid), str(connectivity_gid.hex)])
            lc_conditions = FilterChain(fields=[rm_conditions.fields[0]], operations=[rm_conditions.operations[0]],
                                        values=[rm_conditions.values[0]])
        self.rm = TraitDataTypeSelectField(CortexViewModel.region_mapping_data, name='region_mapping_data',
                                           conditions=rm_conditions)

        self.lc = TraitDataTypeSelectField(CortexViewModel.local_connectivity, name='local_connectivity',
                                           conditions=lc_conditions)
        self.coupling_strength = ArrayField(CortexViewModel.coupling_strength)

    @staticmethod
    def prepare_cortex_fragment(simulator, rendering_rules, form_action_url, project_id):
        surface_index = load_entity_by_gid(simulator.surface.surface_gid)
        form = SimulatorRMFragment(surface_index, simulator.connectivity)
        rm_fragment = AlgorithmService().prepare_adapter_form(form_instance=form, project_id=project_id)
        rm_fragment.fill_from_trait(simulator.surface)

        rendering_rules.form = rm_fragment
        rendering_rules.form_action_url = form_action_url
        return rendering_rules.to_dict()


class SimulatorStimulusFragment(ABCAdapterForm):
    def __init__(self, is_surface_simulation=False):
        super(SimulatorStimulusFragment, self).__init__()
        stimuli_index_class = StimuliRegionIndex
        if is_surface_simulation:
            stimuli_index_class = SpatioTemporalPatternIndex
        traited_field = Attr(stimuli_index_class, doc=SimulatorAdapterModel.stimulus.doc,
                             label=SimulatorAdapterModel.stimulus.label,
                             required=SimulatorAdapterModel.stimulus.required)
        self.stimulus = TraitDataTypeSelectField(traited_field, name='stimulus')

    def fill_trait(self, datatype):
        setattr(datatype, self.stimulus.name, self.stimulus.data)

    def fill_from_trait(self, trait):
        self.stimulus.from_trait(trait, self.stimulus.name)

    @staticmethod
    def prepare_stimulus_fragment(simulator, rendering_rules, is_surface_simulation, form_action_url, project_id):
        form = SimulatorStimulusFragment(is_surface_simulation)
        stimuli_fragment = AlgorithmService().prepare_adapter_form(form_instance=form, project_id=project_id)
        stimuli_fragment.fill_from_trait(simulator)

        rendering_rules.form = stimuli_fragment
        rendering_rules.form_action_url = form_action_url
        return rendering_rules.to_dict()


class SimulatorModelFragment(ABCAdapterForm):
    def __init__(self):
        super(SimulatorModelFragment, self).__init__()
        default_model = ModelsEnum.GENERIC_2D_OSCILLATOR

        self.model = SelectField(
            EnumAttr(default=default_model, label=SimulatorAdapterModel.model.label,
                     doc=SimulatorAdapterModel.model.doc), name='model')

    def fill_from_trait(self, trait):
        # type: (SimulatorAdapterModel) -> None
        self.model.data = type(trait.model)

    def fill_trait(self, datatype):
        if type(datatype.model) != self.model.value:
            datatype.model = self.model.value.instance


class SimulatorIntegratorFragment(ABCAdapterForm):

    def __init__(self):
        super(SimulatorIntegratorFragment, self).__init__()
        self.integrator = SelectField(
            EnumAttr(default=IntegratorViewModelsEnum.HEUN, label=SimulatorAdapterModel.integrator.label,
                     doc=SimulatorAdapterModel.integrator.doc), name='integrator',
            subform=get_form_for_integrator(IntegratorViewModelsEnum.HEUN.value), ui_values=get_integrator_name_list())

    def fill_from_trait(self, trait):
        # type: (SimulatorAdapterModel) -> None
        self.integrator.data = type(trait.integrator)

    def fill_trait(self, datatype):
        datatype.integrator = self.integrator.value.instance


class SimulatorMonitorFragment(ABCAdapterForm):

    def __init__(self, is_surface_simulation=False):
        super(SimulatorMonitorFragment, self).__init__()
        self.monitor_choices = get_ui_name_to_monitor_dict(is_surface_simulation)
        self.is_surface_simulation = is_surface_simulation

        self.monitors = MultiSelectField(List(of=str, label='Monitors', choices=tuple(self.monitor_choices.keys())),
                                         name='monitors')

    def fill_from_trait(self, trait):
        # type: (SimulatorAdapterModel) -> None
        self.monitors.data = [
            get_monitor_to_ui_name_dict(self.is_surface_simulation)[type(monitor)]
            for monitor in trait]


class SimulatorFinalFragment(ABCAdapterForm):

    def __init__(self, default_simulation_name="simulation_1"):
        super(SimulatorFinalFragment, self).__init__()
        self.simulation_length = FloatField(SimulatorAdapterModel.simulation_length)
        self.simulation_name = StrField(Attr(str, doc='Name for the current simulation configuration',
                                             default=default_simulation_name, label='Simulation name'),
                                        name='input_simulation_name_id')

    def fill_from_post(self, form_data):
        super(SimulatorFinalFragment, self).fill_from_post(form_data)
        validation_result = SimulatorFinalFragment.is_burst_name_ok(self.simulation_name.value)
        if validation_result is not True:
            raise ValueError(validation_result)

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

    @staticmethod
    def prepare_final_fragment(simulator, burst_config, project_id, rendering_rules, setup_pse_url):
        simulation_name, simulation_number = BurstService.prepare_simulation_name(burst_config, project_id)
        form = SimulatorFinalFragment(default_simulation_name=simulation_name)
        form.fill_from_trait(simulator)

        rendering_rules.form = form
        rendering_rules.form_action_url = setup_pse_url
        rendering_rules.is_launch_fragment = True
        rendering_rules.is_pse_launch = burst_config.is_pse_burst()
        return rendering_rules.to_dict()


class SimulatorPSEConfigurationFragment(ABCAdapterForm):

    def __init__(self, choices):
        super(SimulatorPSEConfigurationFragment, self).__init__()
        default_choice = choices[0]
        self.pse_param1 = DynamicSelectField(Str(default=default_choice, label="PSE param1"), choices=choices,
                                             name='pse_param1')
        self.pse_param2 = DynamicSelectField(Str(label="PSE param2", required=False), choices=choices,
                                             name='pse_param2')


class SimulatorPSERangeFragment(ABCAdapterForm):
    KEY_PARAM1 = 'param1'
    KEY_PARAM2 = 'param2'
    NAME_FIELD = 'pse_{}_name'
    LO_FIELD = 'pse_{}_lo'
    HI_FIELD = 'pse_{}_hi'
    STEP_FIELD = 'pse_{}_step'
    GID_FIELD = 'pse_{}_guid'

    def __init__(self, pse_param1, pse_param2):
        # type: (RangeParameter, RangeParameter) -> None
        super(SimulatorPSERangeFragment, self).__init__()
        self._add_pse_field(pse_param1)
        if pse_param2:
            self._add_pse_field(pse_param2, self.KEY_PARAM2)

        self.max_pse_number = HiddenField(Int(default=TvbProfile.current.MAX_RANGE_NUMBER, required=False),
                                          "max_range_number")

    def _add_pse_field(self, param, param_key=KEY_PARAM1):
        # type: (RangeParameter, str) -> None
        pse_param_name = HiddenField(Str(default=param.name, required=False), self.NAME_FIELD.format(param_key))
        self.__setattr__(self.NAME_FIELD.format(param_key), pse_param_name)
        if param.type is float:
            self._add_fields_for_float(param, param_key)
        else:
            self._add_field_for_gid(param, param_key)

    def _add_fields_for_float(self, param, param_key):
        # type: (RangeParameter, str) -> None
        pse_param_lo = FloatField(Float(label='LO for {}'.format(param.name), default=param.range_definition.lo,
                                        required=True), name=self.LO_FIELD.format(param_key))
        self.__setattr__(self.LO_FIELD.format(param_key), pse_param_lo)
        pse_param_hi = FloatField(Float(label='HI for {}'.format(param.name), default=param.range_definition.hi,
                                        required=True), name=self.HI_FIELD.format(param_key))
        self.__setattr__(self.HI_FIELD.format(param_key), pse_param_hi)
        pse_param_step = FloatField(Float(label='STEP for {}'.format(param.name), default=param.range_definition.step,
                                          required=True), name=self.STEP_FIELD.format(param_key))
        self.__setattr__(self.STEP_FIELD.format(param_key), pse_param_step)

    def _add_field_for_gid(self, param, param_key):
        # type: (RangeParameter, str) -> None
        traited_attr = Attr(h5.REGISTRY.get_index_for_datatype(param.type), label='Choice for {}'.format(param.name))
        pse_param_dt = TraitDataTypeSelectField(traited_attr, name=self.GID_FIELD.format(param_key),
                                                conditions=param.range_definition, has_all_option=True,
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
        param = BurstService.get_range_param_by_name(param_name, all_range_parameters)
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
