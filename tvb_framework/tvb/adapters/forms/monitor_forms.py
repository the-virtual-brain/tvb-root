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

import numpy

from tvb.adapters.forms.equation_forms import BoldMonitorEquationsEnum
from tvb.basic.neotraits.api import EnumAttr
from tvb.core.entities.file.simulator.view_model import *
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.neotraits.forms import Form, ArrayField, MultiSelectField, FloatField, StrField
from tvb.core.neotraits.forms import SelectField, TraitDataTypeSelectField
from tvb.datatypes.projections import ProjectionsTypeEnum
from tvb.datatypes.sensors import SensorTypesEnum
from tvb.simulator.monitors import DefaultMasks


def get_monitor_to_form_dict():
    monitor_class_to_form = {
        RawViewModel: MonitorForm,
        SubSampleViewModel: MonitorForm,
        SpatialAverageViewModel: SpatialAverageMonitorForm,
        GlobalAverageViewModel: MonitorForm,
        TemporalAverageViewModel: MonitorForm,
        EEGViewModel: EEGMonitorForm,
        MEGViewModel: MEGMonitorForm,
        iEEGViewModel: iEEGMonitorForm,
        BoldViewModel: BoldMonitorForm,
        BoldRegionROIViewModel: BoldMonitorForm
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

    def __init__(self, session_stored_simulator=None, are_params_disabled=False):
        super(MonitorForm, self).__init__()
        self.session_stored_simulator = session_stored_simulator
        self.are_params_disabled = are_params_disabled
        self.period = FloatField(Monitor.period)
        self.variables_of_interest_indexes = {}

        if session_stored_simulator is not None:
            self.variables_of_interest_indexes = session_stored_simulator.determine_indexes_for_chosen_vars_of_interest()

        self.variables_of_interest = MultiSelectField(List(of=str, label='Model Variables to watch',
                                                           choices=tuple(self.variables_of_interest_indexes.keys())),
                                                      name='variables_of_interest')

    def fill_from_trait(self, trait):
        super(MonitorForm, self).fill_from_trait(trait)
        if trait.variables_of_interest is not None:
            self.variables_of_interest.data = [list(self.variables_of_interest_indexes.keys())[idx]
                                               for idx in trait.variables_of_interest]
        else:
            # by default we select all variables of interest for the monitor forms
            self.variables_of_interest.data = list(self.variables_of_interest_indexes.keys())

        if self.are_params_disabled:
            self.period.disabled = True
            self.variables_of_interest.disabled = True

    def fill_trait(self, datatype):
        super(MonitorForm, self).fill_trait(datatype)
        datatype.variables_of_interest = numpy.array(list(self.variables_of_interest_indexes.values()))

    def fill_from_post(self, form_data):
        super(MonitorForm, self).fill_from_post(form_data)
        all_variables = self.session_stored_simulator.model.variables_of_interest
        chosen_variables = form_data['variables_of_interest']
        self.variables_of_interest_indexes = self.session_stored_simulator.\
            get_variables_of_interest_indexes(all_variables, chosen_variables)


class SpatialAverageMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, is_period_disabled=False):
        super(SpatialAverageMonitorForm, self).__init__(session_stored_simulator, is_period_disabled)
        self.spatial_mask = ArrayField(SpatialAverage.spatial_mask)
        self.default_mask = SelectField(SpatialAverage.default_mask)

    def fill_from_trait(self, trait):
        super(SpatialAverageMonitorForm, self).fill_from_trait(trait)
        connectivity_index = load_entity_by_gid(self.session_stored_simulator.connectivity)

        if self.session_stored_simulator.is_surface_simulation is False:
            self.default_mask.choices.remove(DefaultMasks.REGION_MAPPING)

            if connectivity_index.has_cortical_mask is False:
                self.default_mask.choices.remove(DefaultMasks.CORTICAL)

            if connectivity_index.has_hemispheres_mask is False:
                self.default_mask.choices.remove(DefaultMasks.HEMISPHERES)

        else:
            self.default_mask.data = DefaultMasks.REGION_MAPPING
            self.default_mask.disabled = True


class ProjectionMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, is_period_disabled=False):
        super(ProjectionMonitorForm, self).__init__(session_stored_simulator, is_period_disabled)

        rm_filter = None
        if session_stored_simulator and session_stored_simulator.is_surface_simulation:
            rm_filter = FilterChain(fields=[FilterChain.datatype + '.gid'], operations=['=='],
                                    values=[session_stored_simulator.surface.region_mapping_data.hex])

        self.region_mapping = TraitDataTypeSelectField(ProjectionViewModel.region_mapping, name='region_mapping',
                                                       conditions=rm_filter)


class EEGMonitorForm(ProjectionMonitorForm):

    def __init__(self, session_stored_simulator=None, is_period_disabled=False):
        super(EEGMonitorForm, self).__init__(session_stored_simulator, is_period_disabled)

        sensor_filter = FilterChain(fields=[FilterChain.datatype + '.sensors_type'], operations=["=="],
                                    values=[SensorTypesEnum.TYPE_EEG.value])

        projection_filter = FilterChain(fields=[FilterChain.datatype + '.projection_type'], operations=["=="],
                                        values=[ProjectionsTypeEnum.EEG.value])

        self.projection = TraitDataTypeSelectField(EEGViewModel.projection, name='projection',
                                                   conditions=projection_filter)
        self.reference = StrField(EEG.reference)
        self.sensors = TraitDataTypeSelectField(EEGViewModel.sensors, name='sensors', conditions=sensor_filter)
        self.sigma = FloatField(EEG.sigma)


class MEGMonitorForm(ProjectionMonitorForm):

    def __init__(self, session_stored_simulator=None, is_period_disabled=False):
        super(MEGMonitorForm, self).__init__(session_stored_simulator, is_period_disabled)

        sensor_filter = FilterChain(fields=[FilterChain.datatype + '.sensors_type'], operations=["=="],
                                    values=[SensorTypesEnum.TYPE_MEG.value])

        projection_filter = FilterChain(fields=[FilterChain.datatype + '.projection_type'], operations=["=="],
                                        values=[ProjectionsTypeEnum.MEG.value])

        self.projection = TraitDataTypeSelectField(MEGViewModel.projection, name='projection',
                                                   conditions=projection_filter)
        self.sensors = TraitDataTypeSelectField(MEGViewModel.sensors, name='sensors', conditions=sensor_filter)


class iEEGMonitorForm(ProjectionMonitorForm):

    def __init__(self, session_stored_simulator=None, is_period_disabled=False):
        super(iEEGMonitorForm, self).__init__(session_stored_simulator, is_period_disabled)

        sensor_filter = FilterChain(fields=[FilterChain.datatype + '.sensors_type'], operations=["=="],
                                    values=[SensorTypesEnum.TYPE_INTERNAL.value])

        projection_filter = FilterChain(fields=[FilterChain.datatype + '.projection_type'], operations=["=="],
                                        values=[ProjectionsTypeEnum.SEEG.value])

        self.projection = TraitDataTypeSelectField(iEEGViewModel.projection, name='projection',
                                                   conditions=projection_filter)
        self.sigma = FloatField(iEEG.sigma)
        self.sensors = TraitDataTypeSelectField(iEEGViewModel.sensors, name='sensors', conditions=sensor_filter)


class BoldMonitorForm(MonitorForm):

    def __init__(self, session_stored_simulator=None, is_period_disabled=False):
        super(BoldMonitorForm, self).__init__(session_stored_simulator, is_period_disabled)

        self.period = FloatField(Bold.period)
        self.hrf_kernel = SelectField(EnumAttr(label='Equation', default=BoldMonitorEquationsEnum.Gamma_KERNEL),
                                      name='hrf_kernel')

    def fill_trait(self, datatype):
        super(BoldMonitorForm, self).fill_trait(datatype)
        datatype.period = self.period.data
        if type(datatype.hrf_kernel) != self.hrf_kernel.data.value:
            datatype.hrf_kernel = self.hrf_kernel.data.instance

    def fill_from_trait(self, trait):
        super(BoldMonitorForm, self).fill_from_trait(trait)
        self.hrf_kernel.data = trait.hrf_kernel.__class__
