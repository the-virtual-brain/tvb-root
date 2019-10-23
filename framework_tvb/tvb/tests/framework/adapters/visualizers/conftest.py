import pytest
import numpy
from tvb.adapters.datatypes.db import graph
from tvb.adapters.datatypes.db.graph import CovarianceIndex
from tvb.adapters.datatypes.db.mode_decompositions import IndependentComponentsIndex
from tvb.adapters.datatypes.db.spectral import CoherenceSpectrumIndex
from tvb.adapters.datatypes.db.temporal_correlations import CrossCorrelationIndex
from tvb.adapters.datatypes.h5.graph_h5 import CovarianceH5
from tvb.adapters.datatypes.h5.spectral_h5 import CoherenceSpectrumH5
from tvb.adapters.datatypes.h5.temporal_correlations_h5 import CrossCorrelationH5
from tvb.core.entities.storage import dao
from tvb.datatypes import spectral, temporal_correlations
from tvb.core.neocom import h5
from tvb.datatypes.mode_decompositions import IndependentComponents

USER_FULL_NAME = "Datatype Factory User"
DATATYPE_STATE = "RAW_DATA"
DATATYPE_DATA = ["test", "for", "datatypes", "factory"]

DATATYPE_MEASURE_METRIC = {'v': 3}
RANGE_1 = ["row1", [1, 2, 3]]
RANGE_2 = ["row2", [0.1, 0.3, 0.5]]


@pytest.fixture()
def covariance_factory(operation_factory, time_series_index_factory):
    def build():

        time_series_index = time_series_index_factory()
        time_series = h5.load_from_index(time_series_index)
        data = numpy.random.random((10, 10))
        covariance = graph.Covariance(source=time_series, array_data=data)

        op = operation_factory()

        covariance_index = CovarianceIndex()
        covariance_index.fk_from_operation = op.id
        covariance_index.fill_from_has_traits(covariance)

        covariance_h5_path = h5.path_for_stored_index(covariance_index)
        with CovarianceH5(covariance_h5_path) as f:
            f.store(covariance)

        covariance_index = dao.store_entity(covariance_index)
        return covariance_index

    return build

@pytest.fixture()
def cross_coherence_factory(operation_factory, time_series_index_factory):
    def build():
        time_series_index = time_series_index_factory()
        time_series = h5.load_from_index(time_series_index)
        cross_coherence = spectral.CoherenceSpectrum(source=time_series,
                                                    nfft=4,
                                                    array_data=numpy.random.random((10, 10)),
                                                    frequency=numpy.random.random((10,)))

        op = operation_factory()

        cross_coherence_index = CoherenceSpectrumIndex()
        cross_coherence_index.fk_from_operation = op.id
        cross_coherence_index.fill_from_has_traits(cross_coherence)

        cross_coherence_h5_path = h5.path_for_stored_index(cross_coherence_index)
        with CoherenceSpectrumH5(cross_coherence_h5_path) as f:
            f.store(cross_coherence)

        cross_coherence_index = dao.store_entity(cross_coherence_index)
        return cross_coherence_index

    return build


@pytest.fixture()
def cross_correlation_factory(operation_factory):
    def build(time_series):
        data = numpy.random.random((10, 10, 10, 10, 10))
        cross_correlation = temporal_correlations.CrossCorrelation(source=time_series, array_data=data)

        op = operation_factory()

        cross_correlation_index = CrossCorrelationIndex()
        cross_correlation_index.fk_from_operation = op.id
        cross_correlation_index.fill_from_has_traits(cross_correlation)

        cross_correlation_h5_path = h5.path_for_stored_index(cross_correlation_index)
        with CrossCorrelationH5(cross_correlation_h5_path) as f:
            f.store(cross_correlation)

        cross_correlation_index = dao.store_entity(cross_correlation_index)
        return cross_correlation_index

    return build

@pytest.fixture()
def ica_factory(operation_factory, session):
    def build(time_series):
        op = operation_factory()

        ica = IndependentComponents(source=time_series,
                                    component_time_series=numpy.random.random((10, 10, 10, 10)),
                                    prewhitening_matrix=numpy.random.random((10, 10, 10, 10)),
                                    unmixing_matrix=numpy.random.random((10, 10, 10, 10)),
                                    n_components=10)

        ica_index = IndependentComponentsIndex()
        ica_index.fk_from_operation = op.id
        ica_index.fill_from_has_traits(ica)

        # independent_components_h5_path = h5.path_for_stored_index(ica_index)
        # with IndependentComponentsH5(independent_components_h5_path) as f:
        #     f.store(ica)

        session.add(ica_index)
        session.commit()
        return ica_index

    return build