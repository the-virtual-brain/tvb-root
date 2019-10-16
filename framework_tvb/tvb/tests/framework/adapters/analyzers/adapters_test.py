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

import os
from tvb.adapters.analyzers.cross_correlation_adapter import CrossCorrelateAdapter, PearsonCorrelationCoefficientAdapter
from tvb.adapters.analyzers.fcd_adapter import FunctionalConnectivityDynamicsAdapter
from tvb.adapters.analyzers.fmri_balloon_adapter import BalloonModelAdapter
from tvb.adapters.analyzers.ica_adapter import ICAAdapter
from tvb.adapters.analyzers.metrics_group_timeseries import TimeseriesMetricsAdapter
from tvb.adapters.analyzers.node_coherence_adapter import NodeCoherenceAdapter
from tvb.adapters.analyzers.node_complex_coherence_adapter import NodeComplexCoherenceAdapter
from tvb.adapters.analyzers.node_covariance_adapter import NodeCovarianceAdapter
from tvb.adapters.analyzers.pca_adapter import PCAAdapter
from tvb.adapters.analyzers.wavelet_adapter import ContinuousWaveletTransformAdapter
from tvb.adapters.datatypes.h5.fcd_h5 import FcdH5
from tvb.adapters.datatypes.h5.graph_h5 import CovarianceH5, CorrelationCoefficientsH5
from tvb.adapters.datatypes.h5.mapped_value_h5 import DatatypeMeasureH5
from tvb.adapters.datatypes.h5.mode_decompositions_h5 import PrincipalComponentsH5, IndependentComponentsH5
from tvb.adapters.datatypes.h5.spectral_h5 import WaveletCoefficientsH5, CoherenceSpectrumH5, \
    ComplexCoherenceSpectrumH5
from tvb.adapters.datatypes.h5.temporal_correlations_h5 import CrossCorrelationH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.core.neocom import h5
from tvb.tests.framework.adapters.analyzers.fft_test import make_ts_from_op
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class TestAdapters(TransactionalTestCase):
    def test_wavelet_adapter(self, tmpdir, session, operation_factory):
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        wavelet_adapter = ContinuousWaveletTransformAdapter()
        wavelet_adapter.storage_path = storage_folder
        wavelet_adapter.configure(ts_index)

        disk = wavelet_adapter.get_required_disk_size()
        mem = wavelet_adapter.get_required_memory_size()

        wavelet_idx = wavelet_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, WaveletCoefficientsH5, wavelet_idx.gid)
        assert os.path.exists(result_h5)


    def test_pca_adapter(self, tmpdir, session, operation_factory):
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        pca_adapter = PCAAdapter()
        pca_adapter.storage_path = storage_folder
        pca_adapter.configure(ts_index)

        disk = pca_adapter.get_required_disk_size(ts_index)
        mem = pca_adapter.get_required_memory_size(ts_index)

        pca_idx = pca_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, PrincipalComponentsH5, pca_idx.gid)
        assert os.path.exists(result_h5)


    def test_ica_adapter(self, tmpdir, session, operation_factory):
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        ica_adapter = ICAAdapter()
        ica_adapter.storage_path = storage_folder
        ica_adapter.configure(ts_index)

        disk = ica_adapter.get_required_disk_size(ts_index)
        mem = ica_adapter.get_required_memory_size(ts_index)

        ica_idx = ica_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, IndependentComponentsH5, ica_idx.gid)
        assert os.path.exists(result_h5)


    def test_metrics_adapter_launch(self, tmpdir, session, operation_factory):
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        metrics_adapter = TimeseriesMetricsAdapter()
        metrics_adapter.storage_path = storage_folder
        metrics_adapter.configure(ts_index)

        disk = metrics_adapter.get_required_disk_size()
        mem = metrics_adapter.get_required_memory_size()

        datatype_measure_index = metrics_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, DatatypeMeasureH5, datatype_measure_index.gid)
        assert os.path.exists(result_h5)


    def test_cross_correlation_adapter(self, tmpdir, session, operation_factory):
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        cross_correlation_adapter = CrossCorrelateAdapter()
        cross_correlation_adapter.storage_path = storage_folder
        cross_correlation_adapter.configure(ts_index)

        disk = cross_correlation_adapter.get_required_disk_size()
        mem = cross_correlation_adapter.get_required_memory_size()

        cross_correlation_idx = cross_correlation_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, CrossCorrelationH5, cross_correlation_idx.gid)
        assert os.path.exists(result_h5)


    def test_pearson_correlation_coefficient_adapter(self, tmpdir, session, operation_factory):
        # To be fixed once we have the migrated importers
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)
        t_start = 0.9765625
        t_end = 1000.0

        pearson_correlation_coefficient_adapter = PearsonCorrelationCoefficientAdapter()
        pearson_correlation_coefficient_adapter.storage_path = storage_folder
        pearson_correlation_coefficient_adapter.configure(ts_index, t_start, t_end)

        disk = pearson_correlation_coefficient_adapter.get_required_disk_size()
        mem = pearson_correlation_coefficient_adapter.get_required_memory_size()

        correlation_coefficients_idx = pearson_correlation_coefficient_adapter.launch(ts_index, t_start, t_end)

        result_h5 = h5.path_for(storage_folder, CorrelationCoefficientsH5, correlation_coefficients_idx.gid)
        assert os.path.exists(result_h5)


    def test_node_coherence_adapter(self, tmpdir, session, operation_factory):
        # algorithm returns complex values instead of float
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        node_coherence_adapter = NodeCoherenceAdapter()
        node_coherence_adapter.storage_path = storage_folder
        node_coherence_adapter.configure(ts_index)

        disk = node_coherence_adapter.get_required_disk_size()
        mem = node_coherence_adapter.get_required_memory_size()

        coherence_spectrum_idx = node_coherence_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, CoherenceSpectrumH5, coherence_spectrum_idx.gid)
        assert os.path.exists(result_h5)


    def test_node_complex_coherence_adapter(self, tmpdir, session, operation_factory):
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        node_complex_coherence_adapter = NodeComplexCoherenceAdapter()
        node_complex_coherence_adapter.storage_path = storage_folder
        node_complex_coherence_adapter.configure(ts_index)

        disk = node_complex_coherence_adapter.get_required_disk_size()
        mem = node_complex_coherence_adapter.get_required_memory_size()

        complex_coherence_spectrum_idx = node_complex_coherence_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, ComplexCoherenceSpectrumH5, complex_coherence_spectrum_idx.gid)
        assert os.path.exists(result_h5)


    def test_fcd_adapter(self, tmpdir, session, operation_factory):
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)
        sw = 0.5
        sp = 0.2

        fcd_adapter = FunctionalConnectivityDynamicsAdapter()
        fcd_adapter.storage_path = storage_folder
        fcd_adapter.configure(ts_index, sw, sp)

        disk = fcd_adapter.get_required_disk_size()
        mem = fcd_adapter.get_required_memory_size()

        fcd_idx = fcd_adapter.launch(ts_index, sw, sp)

        result_h5 = h5.path_for(storage_folder, FcdH5, fcd_idx.gid)
        assert os.path.exists(result_h5)


    def test_fmri_balloon_adapter(self, tmpdir, session, operation_factory):
        # To be fixed once we have the migrated importers
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        fmri_balloon_adapter = BalloonModelAdapter()
        fmri_balloon_adapter.storage_path = storage_folder
        fmri_balloon_adapter.configure(ts_index)

        disk = fmri_balloon_adapter.get_required_disk_size()
        mem = fmri_balloon_adapter.get_required_memory_size()

        ts_index = fmri_balloon_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, TimeSeriesRegionH5, ts_index.gid)
        assert os.path.exists(result_h5)


    def test_node_covariance_adapter(self, tmpdir, session, operation_factory):
        storage_folder = str(tmpdir)
        ts_index = make_ts_from_op(session, operation_factory)

        node_covariance_adapter = NodeCovarianceAdapter()
        node_covariance_adapter.storage_path = storage_folder
        node_covariance_adapter.configure(ts_index)

        disk = node_covariance_adapter.get_required_disk_size()
        mem = node_covariance_adapter.get_required_memory_size()

        covariance_idx = node_covariance_adapter.launch(ts_index)

        result_h5 = h5.path_for(storage_folder, CovarianceH5, covariance_idx.gid)
        assert os.path.exists(result_h5)
