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

import numpy
from tvb.adapters.simulator.equation_forms import get_ui_name_to_monitor_equation_dict, HRFKernelEquation
from tvb.basic.neotraits.api import List
from tvb.core.entities.file.simulator.view_model import *
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.neotraits.forms import Form, ScalarField, ArrayField, MultiSelectField
from tvb.core.neotraits.forms import SelectField, TraitDataTypeSelectField
from tvb.datatypes.projections import ProjectionsType
from tvb.datatypes.sensors import SensorTypes


def get_monitor_to_form_dict():
    monitor_class_to_form = {
        RawViewModel: RawMonitorForm,
        SubSampleViewModel: SubSampleMonitorForm,
        SpatialAverageViewModel: SpatialAverageMonitorForm,
        GlobalAverageViewModel: GlobalAverageMonitorForm,
        TemporalAverageViewModel: TemporalAverageMonitorForm,
        EEGViewModel: EEGMonitorForm,
        MEGViewModel: MEGMonitorForm,
        iEEGViewModel: iEEGMonitorForm,
        BoldViewModel: BoldMonitorForm,
        BoldRegionROIViewModel: BoldRegionROIMonitorForm
    }

    return monitor_class_to_form


def get_ui_name_to_monitor_dict(surface):
    ui_name_to_monitor = {
        'Raw recording': RawViewModel,
        'Temporally sub-sample': SubSampleViewModel,
        'Spatial average with temporal sub-sample': SpatialAverageViewModel,
        'Global average': GlobalAverageViewModel,
        'Temporal average': TemporalAverageViewModel,
        'EEG': EEGViewModel,
        'MEG': MEGViewModel,
        'Intracerebral / Stereo EEG': iEEGViewModel,
        'BOLD': BoldViewModel
    }

    if surface:
        ui_name_to_monitor['BOLD Region ROI'] = BoldRegionROIViewModel

    return ui_name_to_monitor


def get_monitor_to_ui_name_dict(surface):
    monitor_to_ui_name = dict((v, k) for k, v in get_ui_name_to_monitor_dict(surface).items())
    return monitor_to_ui_name


def get_form_for_monitor(monitor_class):
    return get_monitor_to_form_dict().get(monitor_class)


class MonitorForm(Form):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(MonitorForm, self).__init__(prefix, project_id)
        self.session_stored_simulator = session_stored_simulator
        self.project_id = project_id
        self.period = ScalarField(Monitor.period, self)
        self.variables_of_interest_indexes = {}

        if session_stored_simulator is not None:
            self.variables_of_interest_indexes = self.determine_indexes_for_chosen_vars_of_interest(
                session_stored_simulator)

        self.variables_of_interest = MultiSelectField(List(of=str, label='Model Variables to watch',
                                                           choices=tuple(self.variables_of_interest_indexes.keys())),
                                                      self, name='variables_of_interest')

    def fill_from_trait(self, trait):
        super(MonitorForm, self).fill_from_trait(trait)
        if trait.variables_of_interest is not None:
            self.variables_of_interest.data = [list(self.variables_of_interest_indexes.keys())[idx]
                                               for idx in trait.variables_of_interest]
        else:
            # by default we select all variables of interest for the monitor forms
            self.variables_of_interest.data = list(self.variables_of_interest_indexes.keys())

    def fill_trait(self, datatype):
        super(MonitorForm, self).fill_trait(datatype)
        datatype.variables_of_interest = numpy.array(list(self.variables_of_interest_indexes.values()))

    def fill_from_post(self, form_data):
        super(MonitorForm, self).fill_from_post(form_data)
        all_variables = self.session_stored_simulator.model.variables_of_interest
        chosen_variables = form_data['variables_of_interest']
        self.variables_of_interest_indexes = self._get_variables_of_interest_indexes(all_variables, chosen_variables)

    @staticmethod
    def determine_indexes_for_chosen_vars_of_interest(session_stored_simulator):
        all_variables = session_stored_simulator.model.__class__.variables_of_interest.element_choices
        chosen_variables = session_stored_simulator.model.variables_of_interest
        indexes = MonitorForm._get_variables_of_interest_indexes(all_variables, chosen_variables)
        return indexes

    @staticmethod
    def _get_variables_of_interest_indexes(all_variables, chosen_variables):
        variables_of_interest_indexes = {}

        if not isinstance(chosen_variables, (list, tuple)):
            chosen_variables = [chosen_variables]

        for variable in chosen_variables:
            variables_of_interest_indexes[variable] = all_variables.index(variable)
        return variables_of_interest_indexes

    # TODO: We should review the code here, we could probably reduce the number of  classes that are used here


class RawMonitorForm(Form):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(RawMonitorForm, self).__init__(prefix, project_id)


class SubSampleMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(SubSampleMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)


class SpatialAverageMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(SpatialAverageMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)
        self.spatial_mask = ArrayField(SpatialAverage.spatial_mask, self)
        self.default_mask = ScalarField(SpatialAverage.default_mask, self)

    def fill_from_trait(self, trait):
        super(SpatialAverageMonitorForm, self).fill_from_trait(trait)
        connectivity_index = load_entity_by_gid(self.session_stored_simulator.connectivity)

        if self.session_stored_simulator.is_surface_simulation is False:
            self.default_mask.choices.pop(SpatialAverage.REGION_MAPPING)

            if connectivity_index.has_cortical_mask is False:
                self.default_mask.choices.pop(SpatialAverage.CORTICAL)

            if connectivity_index.has_hemispheres_mask is False:
                self.default_mask.choices.pop(SpatialAverage.HEMISPHERES)

        else:
            self.default_mask.data = SpatialAverage.REGION_MAPPING
            self.default_mask.disabled = True


class GlobalAverageMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(GlobalAverageMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)


class TemporalAverageMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(TemporalAverageMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)


class ProjectionMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(ProjectionMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)

        rm_filter = None
        if session_stored_simulator.is_surface_simulation:
            rm_filter = FilterChain(fields=[FilterChain.datatype + '.gid'], operations=['=='],
                                    values=[session_stored_simulator.surface.region_mapping_data.hex])

        self.region_mapping = TraitDataTypeSelectField(ProjectionViewModel.region_mapping, self,
                                                       name='region_mapping', conditions=rm_filter)


class EEGMonitorForm(ProjectionMonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(EEGMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)

        sensor_filter = FilterChain(fields=[FilterChain.datatype + '.sensors_type'], operations=["=="],
                                    values=[SensorTypes.TYPE_EEG.value])

        projection_filter = FilterChain(fields=[FilterChain.datatype + '.projection_type'], operations=["=="],
                                        values=[ProjectionsType.EEG.value])

        self.projection = TraitDataTypeSelectField(EEGViewModel.projection, self, name='projection',
                                                   conditions=projection_filter)
        self.reference = ScalarField(EEG.reference, self)
        self.sensors = TraitDataTypeSelectField(EEGViewModel.sensors, self, name='sensors', conditions=sensor_filter)
        self.sigma = ScalarField(EEG.sigma, self)


class MEGMonitorForm(ProjectionMonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(MEGMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)

        sensor_filter = FilterChain(fields=[FilterChain.datatype + '.sensors_type'], operations=["=="],
                                    values=[SensorTypes.TYPE_MEG.value])

        projection_filter = FilterChain(fields=[FilterChain.datatype + '.projection_type'], operations=["=="],
                                        values=[ProjectionsType.MEG.value])

        self.projection = TraitDataTypeSelectField(MEGViewModel.projection, self, name='projection',
                                                   conditions=projection_filter)
        self.sensors = TraitDataTypeSelectField(MEGViewModel.sensors, self, name='sensors', conditions=sensor_filter)


class iEEGMonitorForm(ProjectionMonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(iEEGMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)

        sensor_filter = FilterChain(fields=[FilterChain.datatype + '.sensors_type'], operations=["=="],
                                    values=[SensorTypes.TYPE_INTERNAL.value])

        projection_filter = FilterChain(fields=[FilterChain.datatype + '.projection_type'], operations=["=="],
                                        values=[ProjectionsType.SEEG.value])

        self.projection = TraitDataTypeSelectField(iEEGViewModel.projection, self, name='projection',
                                                   conditions=projection_filter)
        self.sigma = ScalarField(iEEG.sigma, self)
        self.sensors = TraitDataTypeSelectField(iEEGViewModel.sensors, self, name='sensors', conditions=sensor_filter)


class BoldMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(BoldMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)
        self.hrf_kernel_choices = get_ui_name_to_monitor_equation_dict()
        default_hrf_kernel = list(self.hrf_kernel_choices.values())[0]

        self.period = ScalarField(Bold.period, self)
        self.hrf_kernel = SelectField(Attr(HRFKernelEquation, label='Equation', default=default_hrf_kernel),
                                      self, name='hrf_kernel', choices=self.hrf_kernel_choices)

    def fill_trait(self, datatype):
        super(BoldMonitorForm, self).fill_trait(datatype)
        datatype.period = self.period.data
        if type(datatype.hrf_kernel) != self.hrf_kernel.data:
            datatype.hrf_kernel = self.hrf_kernel.data()

    def fill_from_trait(self, trait):
        super(BoldMonitorForm, self).fill_from_trait(trait)
        self.hrf_kernel.data = trait.hrf_kernel.__class__


class BoldRegionROIMonitorForm(BoldMonitorForm):

    def __init__(self, session_stored_simulator=None, prefix='', project_id=None):
        super(BoldRegionROIMonitorForm, self).__init__(session_stored_simulator, prefix, project_id)
