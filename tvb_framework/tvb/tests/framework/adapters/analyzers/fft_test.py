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
from tvb.adapters.analyzers.fourier_adapter import FourierAdapter
from tvb.analyzers.fft import compute_fast_fourier_transform
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class TestFFT(TransactionalTestCase):
    def test_fourier_analyser(self, time_series_factory):
        two_node_simple_sin_ts = time_series_factory()

        spectra = compute_fast_fourier_transform(two_node_simple_sin_ts, 4000, None, True)

        # a poor man's peak detection, a box low-pass filter
        peak = 40
        around_peak = spectra.array_data[peak - 10: peak + 10, 0, 0, 0].real
        assert numpy.abs(around_peak).sum() > 0.5 * 20

        peak = 80  # not an expected peak
        around_peak = spectra.array_data[peak - 10: peak + 10, 0, 0, 0].real
        assert numpy.abs(around_peak).sum() < 0.5 * 20

    def test_fourier_adapter(self, time_series_index_factory, operation_from_existing_op_factory):
        # make file stored and indexed time series
        ts_db = time_series_index_factory()

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

        fourier_operation, _ = operation_from_existing_op_factory(ts_db.fk_from_operation)

        adapter = FourierAdapter()
        view_model = adapter.get_view_model_class()()
        view_model.time_series = ts_db.gid
        view_model.segment_length = 400
        adapter.configure(view_model)
        diskq = adapter.get_required_disk_size(view_model)
        memq = adapter.get_required_memory_size(view_model)
        adapter.extract_operation_data(fourier_operation)
        spectra_idx = adapter.launch(view_model)

        assert spectra_idx.fk_source_gid == ts_db.gid
        assert spectra_idx.gid is not None
        assert spectra_idx.segment_length == 1.0  # only 1 sec of signal
