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

import numpy
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5
from tvb.datatypes.time_series import TimeSeries


ntime, nspace, nsv = 120, 4, 2


def make_harmonic_ts():
    return TimeSeries(
        title='harmonic',
        labels_ordering=('time', 'statevar', 'space'),
        labels_dimensions={'statevar': ['position', 'speed']},
        start_time=0.0,
        sample_period=0.5
    )


def harmonic_chunk(time):
    data = numpy.zeros((time.size, nsv, nspace))

    for s in range(nspace):
        data[:, 0, s] = numpy.sin(s*time)
        data[:, 1, s] = numpy.cos(s*time)

    return data


def test_streaming_writes(tmph5factory):
    nchunks = 4
    t = make_harmonic_ts()
    # t.configure will fail for new files, it wants to read the data shape!
    path = tmph5factory()

    with TimeSeriesH5(path) as f:
        # when doing a partial write we still should populate the fields that we can
        # we aim to have the file in the right format even after a chunk write
        # This is similar to super.store but for specific scalars
        f.store(t, scalars_only=True)

        # todo: refuse write_data_slice unless required metadata has been written
        # otherwise we create files that will not conform to the timeseriesh5 format!

        for chunk_id in range(nchunks):
            time = numpy.linspace(chunk_id, chunk_id + 33, ntime, endpoint=False)
            data = harmonic_chunk(time)
            f.write_data_slice(data)


def test_streaming_reads(tmph5factory):
    t = make_harmonic_ts()
    path = tmph5factory()

    with TimeSeriesH5(path) as f:
        time = numpy.linspace(0, 33, ntime)
        data = harmonic_chunk(time)
        f.store(t, scalars_only=True)
        f.write_data_slice(data)

    with TimeSeriesH5(path) as f:
        data = f.read_data_slice((slice(0, 33), slice(None), 0))
        expected = numpy.zeros((33, nsv))
        expected[:, 1] = 1.0   # the cos(0) part
        numpy.testing.assert_array_equal(data, expected)
