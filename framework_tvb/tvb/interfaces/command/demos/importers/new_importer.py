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

"""
Adapter example.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import numpy
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitUploadField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str
from tvb.datatypes.time_series import TimeSeries


class FooDataImporterModel(UploaderViewModel):
    array_data = Str(label='please upload npy', required=True)


class FooDataImporterForm(ABCUploaderForm):

    def __init__(self, project_id=None):
        super(FooDataImporterForm, self).__init__(project_id)
        self.array_data = TraitUploadField(FooDataImporterModel.array_data, '.npy', self.project_id, 'array_data',
                                           self.temporary_files)

    @staticmethod
    def get_view_model():
        return FooDataImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'array_data': '.npy'
        }


class FooDataImporter(ABCUploader):
    _ui_name = "Foo Data"
    _ui_subsection = "foo_data_importer"
    _ui_description = "Foo data import"
    logger = get_logger(__name__)

    def get_form_class(self):
        return FooDataImporterForm

    def get_output(self):
        return [TimeSeriesIndex]

    def launch(self, view_model):
        # type: (FooDataImporterModel) -> TimeSeriesIndex

        array_data = numpy.loadtxt(view_model.array_data)

        ts = TimeSeries(data=array_data)
        ts.configure()

        ts_index = TimeSeriesIndex()
        ts_index.fill_from_has_traits(ts)

        ts_h5_path = h5.path_for(self.storage_path, TimeSeriesH5, ts_index.gid)

        with TimeSeriesH5(ts_h5_path) as ts_h5:
            ts_h5.store(ts, scalars_only=True)
            ts_h5.store_generic_attributes(GenericAttributes())
            ts_h5.write_data_slice(array_data)
        return ts_index
