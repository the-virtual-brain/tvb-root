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
.. moduleauthor:: Marmaduke Woodman <duke@eml.cc>

Provides facilities to import FieldTrip data sets into TVB as
time series and sensor data.

"""

import os
import numpy
import ConfigParser
from scipy.io import loadmat
from StringIO import StringIO
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.time_series import TimeSeriesEEG
from tvb.datatypes.sensors import SensorsEEG


class FieldTripUploader(ABCUploader):
    """
    Upload time series and sensor data via a MAT file containing 
    "dat" and "hdr" variables from the ft_read_data and ft_read_header
    functions.

    For the moment, we treat all data coming from FieldTrip as MEG data though
    the channels may be of heterogeneous type.

    """

    _ui_name = "FieldTrip MAT uploader"
    _ui_subsection = "signals"
    _ui_description = "Upload continuous time-series data from the FieldTrip toolbox"
    logger = get_logger(__name__)


    def get_upload_input_tree(self):
        return [{'name': 'matfile',
                 "type": "upload",
                 'required_type': '.mat',
                 'label': 'Please select a MAT file contain FieldTrip data and header as variables "dat" and "hdr"',
                 'required': 'true'}]


    def get_output(self):
        return [TimeSeriesEEG]


    def launch(self, matfile):
        mat = loadmat(matfile)
        hdr = mat['hdr']
        fs, ns = [hdr[key][0, 0][0, 0] for key in ['Fs', 'nSamples']]

        # the entities to populate
        #ch = SensorsMEG(storage_path=self.storage_path)
        ts = TimeSeriesEEG(storage_path=self.storage_path)

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

        return ts



class VHDR(ABCUploader):
    """
    Upload a BrainVision Analyser file.
    """

    _ui_name = "BrainVision EEG Signal uploader"
    _ui_subsection = "signals"
    _ui_description = "Upload continuous EEG data from a BrainVision file"


    def get_upload_input_tree(self):
        return [{'name': 'vhdr',
                 "type": "upload",
                 'required_type': '.vhdr',
                 'label': 'Please select a VHDR file',
                 'required': 'true'},
                {'name': 'dat',
                 'type': 'upload',
                 'required_type': '.dat',
                 'label': 'Please select the corresponding DAT file',
                 'required': 'true'}]


    def get_output(self):
        return [TimeSeriesEEG]


    def launch(self, vhdr, dat):

        self.filename = vhdr
        self.wd, _ = os.path.split(vhdr)

        # read file
        with open(vhdr, 'r') as fd:
            self.srclines = fd.readlines()

        # config parser expects each section to have header
        # but vhdr has some decorative information at the beginning
        while not self.srclines[0].startswith('['):
            self.srclines.pop(0)

        self.sio = StringIO()
        self.sio.write('\n'.join(self.srclines))
        self.sio.seek(0)

        self.cp = ConfigParser.ConfigParser()
        self.cp.readfp(self.sio)

        for opt in self.cp.options('Common Infos'):
            setattr(self, opt, self.cp.get('Common Infos', opt))

        self.binaryformat = self.cp.get('Binary Infos', 'BinaryFormat')

        self.labels = [self.cp.get('Channel Infos', o).split(',')[0]
                       for o in self.cp.options('Channel Infos')]

        self.fs = self.srate = 1e6 / float(self.samplinginterval)
        self.nchan = int(self.numberofchannels)

        # important if not in same directory
        self.datafile = os.path.join(self.wd, self.datafile)

        self.read_data()

        # create TVB datatypes
        ch = SensorsEEG(
            storage_path=self.storage_path,
            labels=self.labels,
            number_of_sensors=len(self.labels)
        )
        uid = vhdr + '-sensors'
        self._capture_operation_results([ch], uid=uid)

        ts = TimeSeriesEEG(
            sensors=ch,
            storage_path=self.storage_path
        )
        dat = self.data.T[:, numpy.newaxis, :, numpy.newaxis]
        ts.write_data_slice(dat)
        ts.length_1d, ts.length_2d, ts.length_3d, ts.length_4d = dat.shape
        ts.labels_ordering = 'Time 1 Channel 1'.split()
        ts.write_time_slice(numpy.r_[:dat.shape[0]] * 1.0 / self.fs)
        ts.start_time = 0.0
        ts.sample_period_unit = 's'
        ts.sample_period = 1.0 / float(self.fs)
        ts.close_file()

        return ts


    def read_data(self, mmap=False, dt='float32', mode='r'):
        """
        VHDR stores data in channel contiguous way such that reading disparate pieces
        in time is fast, when using memmap.
        """

        if mmap:
            ary = numpy.memmap(self.datafile, dt, mode)
        else:
            ary = numpy.fromfile(self.datafile, dt)
        self.data = ary.reshape((-1, self.nchan)).T
        self.nsamp = self.data.shape[1]



class EEGLAB(ABCUploader):
    """EEGLAB .set file"""

    _ui_name = "EEGLAB .SET uploader"
    _ui_subsection = "signals"
    _ui_description = "Upload continuous time-series data from the EEGLAB toolbox"
    logger = get_logger(__name__)


    def get_upload_input_tree(self):
        return [{'name': 'matfile',
                 "type": "upload",
                 'required_type': '.mat',
                 'label': 'Please select a MAT file contain FieldTrip data and header as variables "dat" and "hdr"',
                 'required': 'true'},
                {'name': 'fdtfile',
                 "type": "upload",
                 'required_type': '.fdt',
                 'label': 'Please select the corresponding FDT file',
                 'required': 'true'}]


    def get_output(self):
        return [TimeSeriesEEG]


    def launch(self, matfile, fdtfile):

        self.mat = loadmat(matfile)
        self.fs = self.mat['EEG']['srate'][0, 0][0, 0]
        self.nsamp = self.mat['EEG']['pnts'][0, 0][0, 0]
        self.data = numpy.fromfile(fdtfile, dtype=numpy.float32)
        self.data = self.data.reshape((self.nsamp, -1)).T
        self.nchan = self.data.shape[0]
        self.labels = [c[0] for c in self.mat['EEG']['chanlocs'][0, 0]['labels'][0]]

        ch = SensorsEEG(
            storage_path=self.storage_path,
            labels=self.labels,
            number_of_sensors=len(self.labels)
        )
        self._capture_operation_results([ch])

        ts = TimeSeriesEEG(
            sensors=ch,
            storage_path=self.storage_path
        )

        # (nchan x ntime) -> (t, sv, ch, mo)
        dat = self.data.T[:, numpy.newaxis, :, numpy.newaxis]

        # write data
        ts.write_data_slice(dat)

        # fill in header info
        ts.length_1d, ts.length_2d, ts.length_3d, ts.length_4d = dat.shape
        ts.labels_ordering = 'Time 1 Channel 1'.split()
        ts.write_time_slice(numpy.r_[:dat.shape[0]] * 1.0 / self.fs)
        ts.start_time = 0.0
        ts.sample_period_unit = 's'
        ts.sample_period = 1.0 / float(self.fs)
        ts.close_file()

        return ts
