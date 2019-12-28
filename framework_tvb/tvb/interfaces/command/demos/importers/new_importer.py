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
from tvb.core.adapters.abcuploader import ABCUploader
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.time_series import TimeSeries


# TODO translate to neoforms
class FooDataImporter(ABCUploader):
    _ui_name = "Foo Data"
    _ui_subsection = "foo_data_importer"
    _ui_description = "Foo data import"
    logger = get_logger(__name__)

    def get_upload_input_tree(self):
        return [
            {'name': 'array_data',
             "type": "upload",
             'required_type': '.npy',
             'label': 'please upload npy',
             'required': 'true'}
        ]

    def get_output(self):
        return [TimeSeriesIndex]

    def launch(self, array_data):

        array_data = numpy.loadtxt(array_data)

        ts = TimeSeries()
        ts.storage_path = self.storage_path
        #ts.configure()
        ts.write_data_slice(array_data)
        ts.close_file()
        return ts


