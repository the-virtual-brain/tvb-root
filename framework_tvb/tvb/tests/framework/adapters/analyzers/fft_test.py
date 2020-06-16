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
import numpy
from tvb.analyzers.fft import FFT
from tvb.adapters.analyzers.fourier_adapter import FourierAdapter
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class TestFFT(TransactionalTestCase):
    def test_fourier_analyser(self, time_series_factory):
        two_node_simple_sin_ts = time_series_factory()

        fft = FFT(time_series=two_node_simple_sin_ts, segment_length=4000.0)
        spectra = fft.evaluate()

        # a poor man's peak detection, a box low-pass filter
        peak = 40
        around_peak = spectra.array_data[peak - 10: peak + 10, 0, 0, 0].real
        assert numpy.abs(around_peak).sum() > 0.5 * 20

        peak = 80  # not an expected peak
        around_peak = spectra.array_data[peak - 10: peak + 10, 0, 0, 0].real
        assert numpy.abs(around_peak).sum() < 0.5 * 20

    def test_fourier_adapter(self, tmpdir, time_series_index_factory):
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

        adapter = FourierAdapter()
        adapter.storage_path = str(tmpdir)
        view_model = adapter.get_view_model_class()()
        view_model.time_series = ts_db.gid
        view_model.segment_length = 400
        adapter.configure(view_model)
        diskq = adapter.get_required_disk_size(view_model)
        memq = adapter.get_required_memory_size(view_model)
        spectra_idx = adapter.launch(view_model)

        assert spectra_idx.fk_source_gid == ts_db.gid
        assert spectra_idx.gid is not None
        assert spectra_idx.segment_length == 1.0  # only 1 sec of signal
