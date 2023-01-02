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

import pytest
import numpy

from tvb.adapters.datatypes.db import graph
from tvb.adapters.datatypes.db.graph import CovarianceIndex
from tvb.adapters.datatypes.db.mode_decompositions import IndependentComponentsIndex
from tvb.adapters.datatypes.db.spectral import CoherenceSpectrumIndex
from tvb.adapters.datatypes.db.temporal_correlations import CrossCorrelationIndex
from tvb.adapters.datatypes.h5.graph_h5 import CovarianceH5
from tvb.adapters.datatypes.h5.mode_decompositions_h5 import IndependentComponentsH5
from tvb.adapters.datatypes.h5.spectral_h5 import CoherenceSpectrumH5
from tvb.adapters.datatypes.h5.temporal_correlations_h5 import CrossCorrelationH5
from tvb.core.entities.storage import dao
from tvb.datatypes import spectral, temporal_correlations
from tvb.core.neocom import h5
from tvb.adapters.datatypes.db import mode_decompositions

USER_FULL_NAME = "Datatype Factory User"
DATATYPE_STATE = "RAW_DATA"
DATATYPE_DATA = ["test", "for", "datatypes", "factory"]

DATATYPE_MEASURE_METRIC = {'v': 3}
RANGE_1 = ["row1", [1, 2, 3]]
RANGE_2 = ["row2", [0.1, 0.3, 0.5]]


@pytest.fixture()
def covariance_factory(time_series_index_factory, operation_factory):
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
def cross_coherence_factory(time_series_index_factory, operation_factory):
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
def cross_correlation_factory(time_series_index_factory, operation_factory):
    def build():
        time_series_index = time_series_index_factory()
        time_series = h5.load_from_index(time_series_index)
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
def ica_factory(operation_factory, time_series_index_factory):
    def build():
        data = numpy.random.random((10, 10, 10, 10))
        time_series_index = time_series_index_factory(data=data)
        time_series = h5.load_from_index(time_series_index)
        n_comp = 5
        ica = mode_decompositions.IndependentComponents(source=time_series,
                                    component_time_series=numpy.random.random((10, n_comp, 10, 10)),
                                    prewhitening_matrix=numpy.random.random((n_comp, 10, 10, 10)),
                                    unmixing_matrix=numpy.random.random((n_comp, n_comp, 10, 10)),
                                    n_components=n_comp)
        ica.compute_norm_source()
        ica.compute_normalised_component_time_series()

        op = operation_factory()

        ica_index = IndependentComponentsIndex()
        ica_index.fk_from_operation = op.id
        ica_index.fill_from_has_traits(ica)

        independent_components_h5_path = h5.path_for_stored_index(ica_index)
        with IndependentComponentsH5(independent_components_h5_path) as f:
            f.store(ica)

        ica_index = dao.store_entity(ica_index)
        return ica_index

    return build
