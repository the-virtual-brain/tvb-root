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

from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.fcd import Fcd
from tvb.datatypes.graph import ConnectivityMeasure, CorrelationCoefficients, Covariance
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.mode_decompositions import PrincipalComponents, IndependentComponents
from tvb.datatypes.patterns import StimuliRegion, StimuliSurface, SpatioTemporalPattern
from tvb.datatypes.projections import ProjectionMatrix
from tvb.datatypes.region_mapping import RegionVolumeMapping, RegionMapping
from tvb.datatypes.sensors import Sensors
from tvb.datatypes.spectral import CoherenceSpectrum, ComplexCoherenceSpectrum, FourierSpectrum, WaveletCoefficients
from tvb.datatypes.structural import StructuralMRI
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.temporal_correlations import CrossCorrelation
from tvb.datatypes.time_series import TimeSeries, TimeSeriesRegion, TimeSeriesSurface, TimeSeriesVolume
from tvb.datatypes.time_series import TimeSeriesEEG, TimeSeriesMEG, TimeSeriesSEEG
from tvb.datatypes.tracts import Tracts
from tvb.datatypes.volumes import Volume
from tvb.datatypes.cortex import Cortex
from tvb.core.entities.file.simulator.cortex_h5 import CortexH5
from tvb.core.entities.file.simulator.simulation_history_h5 import SimulationHistoryH5, SimulationHistory
from tvb.adapters.datatypes.h5.annotation_h5 import ConnectivityAnnotationsH5, ConnectivityAnnotations
from tvb.adapters.datatypes.h5.connectivity_h5 import ConnectivityH5
from tvb.adapters.datatypes.h5.fcd_h5 import FcdH5
from tvb.adapters.datatypes.h5.graph_h5 import ConnectivityMeasureH5, CorrelationCoefficientsH5, CovarianceH5
from tvb.adapters.datatypes.h5.local_connectivity_h5 import LocalConnectivityH5
from tvb.adapters.datatypes.h5.mapped_value_h5 import DatatypeMeasureH5, ValueWrapperH5, ValueWrapper
from tvb.adapters.datatypes.h5.mode_decompositions_h5 import PrincipalComponentsH5, IndependentComponentsH5
from tvb.adapters.datatypes.h5.patterns_h5 import StimuliRegionH5, StimuliSurfaceH5
from tvb.adapters.datatypes.h5.projections_h5 import ProjectionMatrixH5
from tvb.adapters.datatypes.h5.region_mapping_h5 import RegionMappingH5, RegionVolumeMappingH5
from tvb.adapters.datatypes.h5.sensors_h5 import SensorsH5
from tvb.adapters.datatypes.h5.spectral_h5 import CoherenceSpectrumH5, ComplexCoherenceSpectrumH5
from tvb.adapters.datatypes.h5.spectral_h5 import FourierSpectrumH5, WaveletCoefficientsH5
from tvb.adapters.datatypes.h5.structural_h5 import StructuralMRIH5
from tvb.adapters.datatypes.h5.surface_h5 import SurfaceH5
from tvb.adapters.datatypes.h5.temporal_correlations_h5 import CrossCorrelationH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5, TimeSeriesRegionH5, TimeSeriesSurfaceH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesVolumeH5, TimeSeriesEEGH5, TimeSeriesMEGH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesSEEGH5
from tvb.adapters.datatypes.h5.tracts_h5 import TractsH5
from tvb.adapters.datatypes.h5.volumes_h5 import VolumeH5
from tvb.adapters.datatypes.db.annotation import ConnectivityAnnotationsIndex
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.fcd import FcdIndex
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex, CorrelationCoefficientsIndex
from tvb.adapters.datatypes.db.graph import CovarianceIndex
from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex, ValueWrapperIndex
from tvb.adapters.datatypes.db.mode_decompositions import PrincipalComponentsIndex, IndependentComponentsIndex
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex, StimuliSurfaceIndex, SpatioTemporalPatternIndex
from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex, RegionMappingIndex
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.simulation_history import SimulationHistoryIndex
from tvb.adapters.datatypes.db.spectral import CoherenceSpectrumIndex, ComplexCoherenceSpectrumIndex
from tvb.adapters.datatypes.db.spectral import FourierSpectrumIndex, WaveletCoefficientsIndex
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.db.temporal_correlations import CrossCorrelationIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex, TimeSeriesRegionIndex, TimeSeriesSurfaceIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesVolumeIndex, TimeSeriesEEGIndex, TimeSeriesMEGIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesSEEGIndex
from tvb.adapters.datatypes.db.tracts import TractsIndex
from tvb.adapters.datatypes.db.volume import VolumeIndex

from tvb.core.neocom.h5 import REGISTRY


# an alternative approach is to make each h5file declare if it has a corresponding datatype
# then in a metaclass hook each class creation and populate a map
def populate_datatypes_registry():
    REGISTRY.register_datatype(Connectivity, ConnectivityH5, ConnectivityIndex)
    REGISTRY.register_datatype(LocalConnectivity, LocalConnectivityH5, LocalConnectivityIndex)
    REGISTRY.register_datatype(ProjectionMatrix, ProjectionMatrixH5, ProjectionMatrixIndex)
    REGISTRY.register_datatype(RegionVolumeMapping, RegionVolumeMappingH5, RegionVolumeMappingIndex)
    REGISTRY.register_datatype(RegionMapping, RegionMappingH5, RegionMappingIndex)
    REGISTRY.register_datatype(Sensors, SensorsH5, SensorsIndex)
    REGISTRY.register_datatype(SimulationHistory, SimulationHistoryH5, SimulationHistoryIndex)
    REGISTRY.register_datatype(CoherenceSpectrum, CoherenceSpectrumH5, CoherenceSpectrumIndex)
    REGISTRY.register_datatype(ComplexCoherenceSpectrum, ComplexCoherenceSpectrumH5, ComplexCoherenceSpectrumIndex)
    REGISTRY.register_datatype(FourierSpectrum, FourierSpectrumH5, FourierSpectrumIndex)
    REGISTRY.register_datatype(WaveletCoefficients, WaveletCoefficientsH5, WaveletCoefficientsIndex)
    REGISTRY.register_datatype(StructuralMRI, StructuralMRIH5, StructuralMRIIndex)
    REGISTRY.register_datatype(Surface, SurfaceH5, SurfaceIndex)
    REGISTRY.register_datatype(CrossCorrelation, CrossCorrelationH5, CrossCorrelationIndex)
    REGISTRY.register_datatype(TimeSeries, TimeSeriesH5, TimeSeriesIndex)
    REGISTRY.register_datatype(TimeSeriesRegion, TimeSeriesRegionH5, TimeSeriesRegionIndex)
    REGISTRY.register_datatype(TimeSeriesSurface, TimeSeriesSurfaceH5, TimeSeriesSurfaceIndex)
    REGISTRY.register_datatype(TimeSeriesVolume, TimeSeriesVolumeH5, TimeSeriesVolumeIndex)
    REGISTRY.register_datatype(TimeSeriesEEG, TimeSeriesEEGH5, TimeSeriesEEGIndex)
    REGISTRY.register_datatype(TimeSeriesMEG, TimeSeriesMEGH5, TimeSeriesMEGIndex)
    REGISTRY.register_datatype(TimeSeriesSEEG, TimeSeriesSEEGH5, TimeSeriesSEEGIndex)
    REGISTRY.register_datatype(Tracts, TractsH5, TractsIndex)
    REGISTRY.register_datatype(Volume, VolumeH5, VolumeIndex)
    REGISTRY.register_datatype(PrincipalComponents, PrincipalComponentsH5, PrincipalComponentsIndex)
    REGISTRY.register_datatype(IndependentComponents, IndependentComponentsH5, IndependentComponentsIndex)
    REGISTRY.register_datatype(ConnectivityMeasure, ConnectivityMeasureH5, ConnectivityMeasureIndex)
    REGISTRY.register_datatype(CorrelationCoefficients, CorrelationCoefficientsH5, CorrelationCoefficientsIndex)
    REGISTRY.register_datatype(Covariance, CovarianceH5, CovarianceIndex)
    REGISTRY.register_datatype(Fcd, FcdH5, FcdIndex)
    REGISTRY.register_datatype(SpatioTemporalPattern, None, SpatioTemporalPatternIndex)
    REGISTRY.register_datatype(StimuliRegion, StimuliRegionH5, StimuliRegionIndex)
    REGISTRY.register_datatype(StimuliSurface, StimuliSurfaceH5, StimuliSurfaceIndex)
    REGISTRY.register_datatype(None, DatatypeMeasureH5, DatatypeMeasureIndex)
    REGISTRY.register_datatype(ConnectivityAnnotations, ConnectivityAnnotationsH5, ConnectivityAnnotationsIndex)
    REGISTRY.register_datatype(ValueWrapper, ValueWrapperH5, ValueWrapperIndex)
    REGISTRY.register_datatype(Cortex, CortexH5, None)
