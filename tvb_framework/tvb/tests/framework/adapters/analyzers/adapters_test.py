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
from tvb.adapters.datatypes.h5.mode_decompositions_h5 import PrincipalComponentsH5, IndependentComponentsH5
from tvb.adapters.datatypes.h5.spectral_h5 import WaveletCoefficientsH5, CoherenceSpectrumH5, \
    ComplexCoherenceSpectrumH5
from tvb.adapters.datatypes.h5.temporal_correlations_h5 import CrossCorrelationH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.core.entities.file.simulator.datatype_measure_h5 import DatatypeMeasureH5
from tvb.tests.framework.core.base_testcase import TransactionalTestCase


class TestAdapters(TransactionalTestCase):
    def test_wavelet_adapter(self, time_series_index_factory, operation_from_existing_op_factory):
        ts_index = time_series_index_factory()

        wavelet_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        wavelet_adapter = ContinuousWaveletTransformAdapter()
        view_model = wavelet_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        wavelet_adapter.configure(view_model)

        disk = wavelet_adapter.get_required_disk_size(view_model)
        mem = wavelet_adapter.get_required_memory_size(view_model)

        wavelet_adapter.extract_operation_data(wavelet_op)
        wavelet_idx = wavelet_adapter.launch(view_model)

        result_h5 = wavelet_adapter.path_for(WaveletCoefficientsH5, wavelet_idx.gid)
        assert os.path.exists(result_h5)

    def test_pca_adapter(self, time_series_index_factory, operation_from_existing_op_factory):
        ts_index = time_series_index_factory()

        pca_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        pca_adapter = PCAAdapter()
        view_model = pca_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        pca_adapter.configure(view_model)

        disk = pca_adapter.get_required_disk_size(view_model)
        mem = pca_adapter.get_required_memory_size(view_model)

        pca_adapter.extract_operation_data(pca_op)
        pca_idx = pca_adapter.launch(view_model)

        result_h5 = pca_adapter.path_for(PrincipalComponentsH5, pca_idx.gid)
        assert os.path.exists(result_h5)

    def test_ica_adapter(self, time_series_index_factory, operation_from_existing_op_factory):
        ts_index = time_series_index_factory()

        ica_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        ica_adapter = ICAAdapter()
        view_model = ica_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        ica_adapter.configure(view_model)

        disk = ica_adapter.get_required_disk_size(view_model)
        mem = ica_adapter.get_required_memory_size(view_model)

        ica_adapter.extract_operation_data(ica_op)
        ica_idx = ica_adapter.launch(view_model)

        result_h5 = ica_adapter.path_for(IndependentComponentsH5, ica_idx.gid)
        assert os.path.exists(result_h5)

    def test_metrics_adapter_launch(self, time_series_index_factory, operation_from_existing_op_factory):
        ts_index = time_series_index_factory()

        metrics_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        metrics_adapter = TimeseriesMetricsAdapter()
        view_model = metrics_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        metrics_adapter.configure(view_model)

        disk = metrics_adapter.get_required_disk_size(view_model)
        mem = metrics_adapter.get_required_memory_size(view_model)

        metrics_adapter.extract_operation_data(metrics_op)
        datatype_measure_index = metrics_adapter.launch(view_model)

        result_h5 = metrics_adapter.path_for(DatatypeMeasureH5, datatype_measure_index.gid)
        assert os.path.exists(result_h5)

    def test_cross_correlation_adapter(self, time_series_index_factory, operation_from_existing_op_factory):
        ts_index = time_series_index_factory()

        cross_correlation_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        cross_correlation_adapter = CrossCorrelateAdapter()
        view_model = cross_correlation_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        cross_correlation_adapter.configure(view_model)

        disk = cross_correlation_adapter.get_required_disk_size(view_model)
        mem = cross_correlation_adapter.get_required_memory_size(view_model)

        cross_correlation_adapter.extract_operation_data(cross_correlation_op)
        cross_correlation_idx = cross_correlation_adapter.launch(view_model)

        result_h5 = cross_correlation_adapter.path_for(CrossCorrelationH5, cross_correlation_idx.gid)
        assert os.path.exists(result_h5)

    def test_pearson_correlation_coefficient_adapter(self, time_series_index_factory,
                                                     operation_from_existing_op_factory):
        # To be fixed once we have the migrated importers
        ts_index = time_series_index_factory()

        pearson_correlation_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        pearson_correlation_coefficient_adapter = PearsonCorrelationCoefficientAdapter()
        view_model = pearson_correlation_coefficient_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        pearson_correlation_coefficient_adapter.configure(view_model)

        disk = pearson_correlation_coefficient_adapter.get_required_disk_size(view_model)
        mem = pearson_correlation_coefficient_adapter.get_required_memory_size(view_model)

        pearson_correlation_coefficient_adapter.extract_operation_data(pearson_correlation_op)
        correlation_coefficients_idx = pearson_correlation_coefficient_adapter.launch(view_model)

        result_h5 = pearson_correlation_coefficient_adapter.path_for(CorrelationCoefficientsH5,
                                                                     correlation_coefficients_idx.gid)
        assert os.path.exists(result_h5)

    def test_node_coherence_adapter(self, time_series_index_factory, operation_from_existing_op_factory):
        # algorithm returns complex values instead of float
        ts_index = time_series_index_factory()

        node_coherence_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        node_coherence_adapter = NodeCoherenceAdapter()
        view_model = node_coherence_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        node_coherence_adapter.configure(view_model)

        disk = node_coherence_adapter.get_required_disk_size(view_model)
        mem = node_coherence_adapter.get_required_memory_size(view_model)

        node_coherence_adapter.extract_operation_data(node_coherence_op)
        coherence_spectrum_idx = node_coherence_adapter.launch(view_model)

        result_h5 = node_coherence_adapter.path_for(CoherenceSpectrumH5, coherence_spectrum_idx.gid)
        assert os.path.exists(result_h5)

    def test_node_complex_coherence_adapter(self, time_series_index_factory, operation_from_existing_op_factory):
        ts_index = time_series_index_factory()

        complex_coherence_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        node_complex_coherence_adapter = NodeComplexCoherenceAdapter()
        view_model = node_complex_coherence_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        node_complex_coherence_adapter.configure(view_model)

        disk = node_complex_coherence_adapter.get_required_disk_size(view_model)
        mem = node_complex_coherence_adapter.get_required_memory_size(view_model)

        node_complex_coherence_adapter.extract_operation_data(complex_coherence_op)
        complex_coherence_spectrum_idx = node_complex_coherence_adapter.launch(view_model)

        result_h5 = node_complex_coherence_adapter.path_for(ComplexCoherenceSpectrumH5,
                                                            complex_coherence_spectrum_idx.gid)
        assert os.path.exists(result_h5)

    def test_fcd_adapter(self, time_series_region_index_factory, connectivity_index_factory,
                         connectivity_factory, region_mapping_factory, surface_factory,
                         operation_from_existing_op_factory):
        connectivity = connectivity_factory()
        connectivity_index_factory(conn=connectivity)
        surface = surface_factory()
        region_mapping = region_mapping_factory(surface=surface, connectivity=connectivity)
        ts_index = time_series_region_index_factory(connectivity=connectivity, region_mapping=region_mapping)

        fcd_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        fcd_adapter = FunctionalConnectivityDynamicsAdapter()
        view_model = fcd_adapter.get_view_model_class()()
        view_model.sw = 0.5
        view_model.sp = 0.2
        view_model.time_series = ts_index.gid
        fcd_adapter.configure(view_model)

        disk = fcd_adapter.get_required_disk_size(view_model)
        mem = fcd_adapter.get_required_memory_size(view_model)

        fcd_adapter.extract_operation_data(fcd_op)
        fcd_idx = fcd_adapter.launch(view_model)

        result_h5 = fcd_adapter.path_for(FcdH5, fcd_idx[0].gid)
        assert os.path.exists(result_h5)

    def test_fmri_balloon_adapter(self, time_series_region_index_factory,
                                  connectivity_factory, region_mapping_factory, surface_factory,
                                  operation_from_existing_op_factory):
        connectivity = connectivity_factory()
        surface = surface_factory()
        region_mapping = region_mapping_factory(surface=surface, connectivity=connectivity)
        ts_index = time_series_region_index_factory(connectivity=connectivity, region_mapping=region_mapping)

        fmri_balloon_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        fmri_balloon_adapter = BalloonModelAdapter()
        view_model = fmri_balloon_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        fmri_balloon_adapter.configure(view_model)

        disk = fmri_balloon_adapter.get_required_disk_size(view_model)
        mem = fmri_balloon_adapter.get_required_memory_size(view_model)
        assert disk > 0
        assert mem > 0

        fmri_balloon_adapter.extract_operation_data(fmri_balloon_op)
        ts_index = fmri_balloon_adapter.launch(view_model)

        result_h5 = fmri_balloon_adapter.path_for(TimeSeriesRegionH5, ts_index.gid)
        assert os.path.exists(result_h5)

    def test_node_covariance_adapter(self, time_series_index_factory, operation_from_existing_op_factory):
        ts_index = time_series_index_factory()

        node_covariance_op, project_id = operation_from_existing_op_factory(ts_index.fk_from_operation)

        node_covariance_adapter = NodeCovarianceAdapter()
        view_model = node_covariance_adapter.get_view_model_class()()
        view_model.time_series = ts_index.gid
        node_covariance_adapter.configure(view_model)

        disk = node_covariance_adapter.get_required_disk_size(view_model)
        mem = node_covariance_adapter.get_required_memory_size(view_model)

        node_covariance_adapter.extract_operation_data(node_covariance_op)
        covariance_idx = node_covariance_adapter.launch(view_model)

        result_h5 = node_covariance_adapter.path_for(CovarianceH5, covariance_idx.gid)
        assert os.path.exists(result_h5)
