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
import numpy
from tvb.analyzers.fft import FFT
from tvb.datatypes.time_series import TimeSeries
from tvb.core.entities.file.datatypes.time_series_h5 import TimeSeriesH5
from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex
from tvb.core.neocom import h5
from tvb.adapters.analyzers.fourier_adapter import FourierAdapter


def make_ts():
    # 1 second time series, 4000 samples
    # channel 0: a pure 40hz
    # channel 1: a pure 200hz
    # channel 2: a superposition of 100 and 300Hz equal amplitude
    time = numpy.linspace(0, 1000, 4000)
    data = numpy.zeros((time.size, 1, 3, 1))
    data[:, 0, 0, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 40)
    data[:, 0, 1, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 200)
    data[:, 0, 2, 0] = numpy.sin(2 * numpy.pi * time / 1000.0 * 100) + \
                       numpy.sin(2 * numpy.pi * time / 1000.0 * 300)

    return TimeSeries(time=time, data=data, sample_period=1.0 / 4000)


def make_ts_from_op(session, operationFactory):
    # make file stored and indexed time series
    two_node_simple_sin_ts = make_ts()
    op = operationFactory()

    ts_db = TimeSeriesIndex()
    ts_db.fk_from_operation = op.id
    ts_db.fill_from_has_traits(two_node_simple_sin_ts)

    ts_h5_path = h5.path_for_stored_index(ts_db)
    with TimeSeriesH5(ts_h5_path) as f:
        f.store(two_node_simple_sin_ts)
        f.sample_rate.store(two_node_simple_sin_ts.sample_rate)

    session.add(ts_db)
    session.commit()
    return ts_db


def test_fourier_analyser():
    two_node_simple_sin_ts = make_ts()

    fft = FFT(time_series=two_node_simple_sin_ts, segment_length=4000.0)
    spectra = fft.evaluate()

    # a poor man's peak detection, a box low-pass filter
    peak = 40
    around_peak = spectra.array_data[peak - 10: peak + 10, 0, 0, 0].real
    assert numpy.abs(around_peak).sum() > 0.5 * 20

    peak = 80  # not an expected peak
    around_peak = spectra.array_data[peak - 10: peak + 10, 0, 0, 0].real
    assert numpy.abs(around_peak).sum() < 0.5 * 20


def test_fourier_adapter(tmpdir, session, operationFactory):
    # make file stored and indexed time series
    ts_db = make_ts_from_op(session, operationFactory)

    # we have the required data to start the adapter
    # REVIEW THIS
    # The adapter methods require the shape of the full time series
    # As we do not want to load the whole time series we can not send the
    # datatype. What the adapter actually wants is metadata about the datatype
    # on disk. So the database entity is a good input.
    # But this contradicts the convention that adapters inputs are datatypes
    # Note that the previous functionality relied on the fact that datatypes
    # knew directly how to interact with storage via get_data_shape
    # Another interpretation is that the adapter really wants a data type
    # weather it is a in memory HasTraits or a H5File to access it.
    # Here we consider this last option.

    adapter = FourierAdapter()
    adapter.storage_path = str(tmpdir)
    adapter.configure(ts_db, segment_length=400)
    diskq = adapter.get_required_disk_size(ts_db, segment_length=400)
    memq = adapter.get_required_memory_size(ts_db, segment_length=400)
    spectra_idx = adapter.launch(ts_db, segment_length=400)

    assert spectra_idx.source_gid == ts_db.gid
    assert spectra_idx.gid is not None
    assert spectra_idx.segment_length == 1.0  # only 1 sec of signal
