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

from tvb.adapters.uploaders.mat_timeseries_importer import RegionMatTimeSeriesImporterForm, TS_EEG, \
    RegionTimeSeriesImporter, RegionMatTimeSeriesImporterModel
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import DataTypeGidAttr
from tvb.datatypes.sensors import Sensors, SensorTypesEnum


class EEGMatTimeSeriesImporterModel(RegionMatTimeSeriesImporterModel):
    datatype = DataTypeGidAttr(
        linked_datatype=Sensors,
        label='EEG Sensors'
    )


class EEGRegionMatTimeSeriesImporterForm(RegionMatTimeSeriesImporterForm):

    def __init__(self):
        super(EEGRegionMatTimeSeriesImporterForm, self).__init__()
        eeg_conditions = FilterChain(fields=[FilterChain.datatype + '.sensors_type'], operations=['=='],
                                     values=[SensorTypesEnum.TYPE_EEG.value])
        self.datatype = TraitDataTypeSelectField(EEGMatTimeSeriesImporterModel.datatype, name='tstype_parameters',
                                                 conditions=eeg_conditions)

    @staticmethod
    def get_view_model():
        return EEGMatTimeSeriesImporterModel


class EEGRegionTimeSeriesImporter(RegionTimeSeriesImporter):
    _ui_name = "TimeSeries EEG MAT"
    tstype = TS_EEG

    def get_form_class(self):
        return EEGRegionMatTimeSeriesImporterForm
