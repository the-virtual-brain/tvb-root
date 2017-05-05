# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

Provides facilities to import FieldTrip data sets into TVB as time series and sensor data.

"""

import numpy
import scipy.io
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.time_series import TimeSeries
from tvb.datatypes.sensors import SensorsMEG as Sensors



class FieldTripUploader(ABCUploader):
    """
    Upload time series and sensor data via a MAT file containing 
    "dat" and "hdr" variables from the ft_read_data and ft_read_header
    functions.

    For the moment, we treat all data coming from FieldTrip as MEG data though
    the channels may be of heterogeneous type.

    """

    _ui_name = "FieldTrip"
    _ui_subsection = "fieldtrip_upload"
    _ui_description = "Upload continuous time-series data from the FieldTrip toolbox"
    logger = get_logger(__name__)


    def get_upload_input_tree(self):
        return [{'name': 'matfile',
                 "type": "upload",
                 'required_type': '.mat',
                 'label': 'Please select a MAT file contain FieldTrip data and header as variables "dat" and "hdr"',
                 'required': 'true'}]


    def get_output(self):
        return [TimeSeries]#, Sensors]


    def launch(self, matfile):
        mat = scipy.io.loadmat(matfile)
        hdr = mat['hdr']
        fs, ns = [hdr[key][0, 0][0, 0] for key in ['Fs', 'nSamples']]

        # the entities to populate
        #ch = Sensors(storage_path=self.storage_path)
        ts = TimeSeries(storage_path=self.storage_path)

        # (nchan x ntime) -> (t, sv, ch, mo)
        dat = mat['dat'].T[:, numpy.newaxis, :, numpy.newaxis]

        # write data
        ts.write_data_slice(dat)

        # fill in header info
        ts.length_1d, ts.length_2d, ts.length_3d, ts.length_4d = dat.shape
        ts.labels_ordering = 'Time 1 Channel 1'.split()
        ts.write_time_slice(numpy.r_[:ns] * 1.0 / fs)
        ts.start_time = 0.0
        ts.sample_period_unit = 's'
        ts.sample_period = 1.0 / float(fs)
        ts.close_file()

        # setup sensors information

        # ch.labels = numpy.array(
        #     [str(l[0]) for l in hdr['label'][0, 0][:, 0]])
        # ch.number_of_sensors = ch.labels.size

        return ts #, ch
