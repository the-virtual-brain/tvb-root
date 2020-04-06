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
from tvb.core.entities.filters.chain import FilterChain
from tvb.basic.neotraits.api import Attr, Range
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.surfaces import CORTICAL
from tvb.simulator.simulator import Simulator
from tvb.adapters.simulator.integrator_forms import get_ui_name_to_integrator_dict
from tvb.adapters.simulator.model_forms import get_ui_name_to_model
from tvb.adapters.simulator.monitor_forms import get_ui_name_to_monitor_dict
from tvb.adapters.simulator.range_parameter import RangeParameter
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.datatypes.db.patterns import StimuliSurfaceIndex, StimuliRegionIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.neotraits.forms import DataTypeSelectField, SimpleSelectField, ScalarField, ArrayField, SimpleFloatField, \
    SimpleHiddenField
from tvb.core.neocom import h5


class SimulatorSurfaceFragment(ABCAdapterForm):
    def __init__(self, prefix='', project_id=None):
        super(SimulatorSurfaceFragment, self).__init__(prefix, project_id)
        conditions = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=["=="],
                                 values=[CORTICAL])
        self.surface = DataTypeSelectField(SurfaceIndex, self, name='surface', required=False,
                                           label=Simulator.surface.label, doc=Simulator.surface.doc, conditions=conditions)

    def fill_from_trait(self, trait):
        # type: (Simulator) -> None
        if trait.surface:
            if hasattr(trait.surface, 'surface_gid'):
                self.surface.data = trait.surface.surface_gid.hex
        else:
            self.surface.data = None


class SimulatorRMFragment(ABCAdapterForm):
    def __init__(self, prefix='', project_id=None, surface_index=None):
        super(SimulatorRMFragment, self).__init__(prefix, project_id)
        conditions = None
        if surface_index:
            conditions = FilterChain(fields=[FilterChain.datatype + '.surface_gid'], operations=["=="],
                                     values=[str(surface_index.gid)])
        self.rm = DataTypeSelectField(RegionMappingIndex, self, name='region_mapping', required=True,
                                      label=Cortex.region_mapping_data.label,
                                      doc=Cortex.region_mapping_data.doc, conditions=conditions)
        self.lc = DataTypeSelectField(LocalConnectivityIndex, self, name='local_connectivity',
                                      label=Cortex.local_connectivity.label, doc=Cortex.local_connectivity.doc,
                                      conditions=conditions)
        self.coupling_strength = ArrayField(Cortex.coupling_strength, self)

    def fill_from_trait(self, trait):
        # type: (Simulator) -> None
        self.coupling_strength.data = trait.surface.coupling_strength
        if hasattr(trait.surface, 'region_mapping_data'):
            self.rm.data = trait.surface.region_mapping_data.hex
        else:
            self.rm.data = None
        if trait.surface.local_connectivity:
            self.lc.data = trait.surface.local_connectivity.hex
        else:
            self.lc.data = None


class SimulatorStimulusFragment(ABCAdapterForm):
    def __init__(self, prefix='', project_id=None, is_surface_simulation=False):
        super(SimulatorStimulusFragment, self).__init__(prefix, project_id)
        if is_surface_simulation:
            stimuli_index_class = StimuliSurfaceIndex
        else:
            stimuli_index_class = StimuliRegionIndex
        self.stimulus = DataTypeSelectField(stimuli_index_class, self, name='region_stimuli', required=False,
                                            label=Simulator.stimulus.label, doc=Simulator.stimulus.doc)

    def fill_from_trait(self, trait):
        # type: (Simulator) -> None
        if hasattr(trait, 'stimulus') and trait.stimulus is not None:
            self.stimulus.data = trait.stimulus.hex
        else:
            self.stimulus.data = None


class SimulatorModelFragment(ABCAdapterForm):
    def __init__(self, prefix='', project_id=None):
        super(SimulatorModelFragment, self).__init__(prefix, project_id)

        self.model_choices = get_ui_name_to_model()

        self.model = SimpleSelectField(choices=self.model_choices, form=self, name='model', required=True,
                                       label=Simulator.model.label, doc=Simulator.model.doc)
        self.model.template = "form_fields/select_field.html"

    def fill_from_trait(self, trait):
        # type: (Simulator) -> None
        self.model.data = trait.model.__class__


class SimulatorIntegratorFragment(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(SimulatorIntegratorFragment, self).__init__(prefix, project_id)

        self.integrator_choices = get_ui_name_to_integrator_dict()

        self.integrator = SimpleSelectField(choices=self.integrator_choices, form=self, name='integrator',
                                            required=True,
                                            label=Simulator.integrator.label, doc=Simulator.integrator.doc)
        self.integrator.template = "form_fields/select_field.html"

    def fill_from_trait(self, trait):
        # type: (Simulator) -> None
        self.integrator.data = trait.integrator.__class__


class SimulatorMonitorFragment(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None, is_surface_simulation=False):
        super(SimulatorMonitorFragment, self).__init__(prefix, project_id)

        self.monitor_choices = get_ui_name_to_monitor_dict(is_surface_simulation)

        self.monitor = SimpleSelectField(choices=self.monitor_choices, form=self, name='monitor', required=True,
                                         label=Simulator.monitors.label, doc=Simulator.monitors.doc)
        self.monitor.template = "form_fields/select_field.html"

    def fill_from_trait(self, trait):
        # type: (Simulator) -> None
        self.monitor.data = trait.monitors[0].__class__


class SimulatorFinalFragment(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None, simulation_number=1):
        super(SimulatorFinalFragment, self).__init__(prefix, project_id)
        default_simulation_name = "simulation_" + str(simulation_number)
        self.simulation_length = ScalarField(Simulator.simulation_length, self)
        self.simulation_name = ScalarField(Attr(str, doc='Name for the current simulation configuration', default=default_simulation_name,
                                                label='Simulation name'), self, name='input_simulation_name_id')

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
        self.pse_param1 = SimpleSelectField(choices, form=self, name='pse_param1', required=True, label="PSE param1")
        self.pse_param1.template = "form_fields/select_field.html"
        self.pse_param2 = SimpleSelectField(choices, form=self, name='pse_param2', label="PSE param2")
        self.pse_param2.template = "form_fields/select_field.html"


class SimulatorPSEParamRangeFragment(ABCAdapterForm):

    def __init__(self, pse_param1, pse_param2, prefix='', project_id=None):
        # type: (RangeParameter) -> None
        super(SimulatorPSEParamRangeFragment, self).__init__(prefix, project_id)
        self.pse_param1_name = SimpleHiddenField(self, name='pse_param1_name', default=pse_param1.name)
        if pse_param1.type is float:
            self.pse_param1_lo = SimpleFloatField(self, name='pse_param1_lo', required=True, label='PSE param1 lo',
                                                  default=pse_param1.range_definition.lo)
            self.pse_param1_hi = SimpleFloatField(self, name='pse_param1_hi', required=True, label='PSE param1 hi',
                                                  default=pse_param1.range_definition.hi)
            self.pse_param1_step = SimpleFloatField(self, name='pse_param1_step', required=True,
                                                    label='PSE param1 step', default=pse_param1.range_definition.step)

        else:
            self.pse_param1_dt = DataTypeSelectField(h5.REGISTRY.get_index_for_datatype(pse_param1.type), self,
                                                     name='pse_param1_guid', required=True, label='PSE param1 guid',
                                                     dynamic_conditions=pse_param1.range_definition,
                                                     has_all_option=True)

        if pse_param2:
            self.pse_param2_name = SimpleHiddenField(self, name='pse_param2_name', default=pse_param2.name)
            if pse_param2.type is float:
                self.pse_param2_lo = SimpleFloatField(self, name='pse_param2_lo', required=True, label='PSE param2 lo',
                                                      default=pse_param2.range_definition.lo)
                self.pse_param2_hi = SimpleFloatField(self, name='pse_param2_hi', required=True, label='PSE param2 hi',
                                                      default=pse_param2.range_definition.hi)
                self.pse_param2_step = SimpleFloatField(self, name='pse_param2_step', required=True,
                                                        label='PSE param2 step',
                                                        default=pse_param2.range_definition.step)
            else:
                self.pse_param2_dt = DataTypeSelectField(h5.REGISTRY.get_index_for_datatype(pse_param2.type), self,
                                                         name='pse_param2_guid', required=True, label='PSE param2 guid',
                                                         dynamic_conditions=pse_param2.range_definition,
                                                         has_all_option=True)

    @staticmethod
    def _prepare_pse_uuid_list(pse_uuid_str):
        pse_uuid_str_list = pse_uuid_str.split(',')
        pse_uuid_list = [uuid.UUID(uuid_str) for uuid_str in pse_uuid_str_list]
        return pse_uuid_list

    @staticmethod
    def fill_from_post(all_range_parameters, **data):
        pse_param1_name = data.get('pse_param1_name')
        pse_param2_name = data.get('pse_param2_name')

        pse_param1 = all_range_parameters.get(pse_param1_name)
        if pse_param1.type is float:
            pse_param1_lo = data.get('pse_param1_lo')
            pse_param1_hi = data.get('pse_param1_hi')
            pse_param1_step = data.get('pse_param1_step')
            param1_range = RangeParameter(pse_param1_name, pse_param1.type,
                                          Range(float(pse_param1_lo), float(pse_param1_hi), float(pse_param1_step)),
                                          is_array=pse_param1.is_array)
        else:
            param1_range_str = data.get('pse_param1_guid')
            param1_range = RangeParameter(pse_param1_name, pse_param1.type,
                                          SimulatorPSEParamRangeFragment._prepare_pse_uuid_list(param1_range_str))

        param2_range = None
        if pse_param2_name:
            pse_param2 = all_range_parameters.get(pse_param2_name)
            if pse_param2.type is float:
                pse_param2_lo = data.get('pse_param2_lo')
                pse_param2_hi = data.get('pse_param2_hi')
                pse_param2_step = data.get('pse_param2_step')
                param2_range = RangeParameter(pse_param2_name, float,
                                              Range(float(pse_param2_lo), float(pse_param2_hi), float(pse_param2_step)),
                                              is_array=pse_param2.is_array)
            else:
                param2_range_str = data.get('pse_param2_guid')
                param2_range = RangeParameter(pse_param2_name, pse_param2.type,
                                              SimulatorPSEParamRangeFragment._prepare_pse_uuid_list(param2_range_str))

        return param1_range, param2_range

