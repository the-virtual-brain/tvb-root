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

from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import DataTypeGidAttr
from tvb.datatypes.sensors import Sensors, SensorTypes
from tvb.adapters.uploaders.mat_timeseries_importer import RegionMatTimeSeriesImporterForm, TS_EEG, \
    RegionTimeSeriesImporter
from tvb.core.neotraits.forms import TraitDataTypeSelectField


class EEGMatTimeSeriesImporterModel(UploaderViewModel):
    datatype = DataTypeGidAttr(
        linked_datatype=Sensors,
        label='EEG Sensors'
    )


class EEGRegionMatTimeSeriesImporterForm(RegionMatTimeSeriesImporterForm):

    def __init__(self, project_id=None):
        super(EEGRegionMatTimeSeriesImporterForm, self).__init__(project_id)
        eeg_conditions = FilterChain(fields=[FilterChain.datatype + '.sensors_type'], operations=['=='],
                                     values=[SensorTypes.TYPE_EEG.value])
        self.datatype = TraitDataTypeSelectField(EEGMatTimeSeriesImporterModel.datatype, self.project_id,
                                                 name='tstype_parameters', conditions=eeg_conditions)

    @staticmethod
    def get_view_model():
        return EEGMatTimeSeriesImporterModel


class EEGRegionTimeSeriesImporter(RegionTimeSeriesImporter):
    _ui_name = "Timeseries EEG MAT"
    tstype = TS_EEG

    def get_form_class(self):
        return EEGRegionMatTimeSeriesImporterForm
