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
from tvb.basic.profile import TvbProfile
from tvb.datatypes.simulation_state import SimulationState

from tvb.adapters.analyzers.analyze_algorithm_category_config import AnalyzeAlgorithmCategoryConfig
from tvb.adapters.analyzers.cross_correlation_adapter import CrossCorrelateAdapter, PearsonCorrelationCoefficientAdapter
from tvb.adapters.analyzers.fcd_adapter import FunctionalConnectivityDynamicsAdapter
from tvb.adapters.analyzers.fmri_balloon_adapter import BalloonModelAdapter
from tvb.adapters.analyzers.fourier_adapter import FourierAdapter
from tvb.adapters.analyzers.ica_adapter import ICAAdapter
from tvb.adapters.analyzers.metrics_group_timeseries import TimeseriesMetricsAdapter
from tvb.adapters.analyzers.node_coherence_adapter import NodeCoherenceAdapter
from tvb.adapters.analyzers.node_complex_coherence_adapter import NodeComplexCoherenceAdapter
from tvb.adapters.analyzers.node_covariance_adapter import NodeCovarianceAdapter
from tvb.adapters.analyzers.pca_adapter import PCAAdapter
from tvb.adapters.analyzers.wavelet_adapter import ContinuousWaveletTransformAdapter
from tvb.adapters.creators.allen_creator import AllenConnectomeBuilder
from tvb.adapters.creators.connectivity_creator import ConnectivityCreator
from tvb.adapters.creators.create_algorithm_category_config import CreateAlgorithmCategoryConfig
from tvb.adapters.creators.local_connectivity_creator import LocalConnectivityCreator
from tvb.adapters.creators.stimulus_creator import RegionStimulusCreator, SurfaceStimulusCreator
from tvb.adapters.simulator.simulate_algorithm_category_config import SimulateAlgorithmCategoryConfig
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter
from tvb.adapters.uploaders.brco_importer import BRCOImporter
from tvb.adapters.uploaders.cff_importer import CFF_Importer
from tvb.adapters.uploaders.connectivity_measure_importer import ConnectivityMeasureImporter
from tvb.adapters.uploaders.csv_connectivity_importer import CSVConnectivityImporter
from tvb.adapters.uploaders.eeg_mat_timeseries_importer import EEGMatTimeSeriesImporter
from tvb.adapters.uploaders.gifti_surface_importer import GIFTISurfaceImporter
from tvb.adapters.uploaders.gifti_timeseries_importer import GIFTITimeSeriesImporter
from tvb.adapters.uploaders.mat_timeseries_importer import MatTimeSeriesImporter
from tvb.adapters.uploaders.networkx_importer import NetworkxConnectivityImporter
from tvb.adapters.uploaders.nifti_importer import NIFTIImporter
from tvb.adapters.uploaders.obj_importer import ObjSurfaceImporter
from tvb.adapters.uploaders.projection_matrix_importer import ProjectionMatrixSurfaceEEGImporter
from tvb.adapters.uploaders.region_mapping_importer import RegionMapping_Importer
from tvb.adapters.uploaders.sensors_importer import Sensors_Importer
from tvb.adapters.uploaders.tract_importer import TrackvizTractsImporter, ZipTxtTractsImporter
from tvb.adapters.uploaders.tvb_importer import TVBImporter
from tvb.adapters.uploaders.upload_algorithm_category_config import UploadAlgorithmCategoryConfig
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporter
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporter
from tvb.adapters.visualizers.annotations_viewer import ConnectivityAnnotationsView
from tvb.adapters.visualizers.brain import BrainViewer, DualBrainViewer
from tvb.adapters.visualizers.complex_imaginary_coherence import ImaginaryCoherenceDisplay
from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.adapters.visualizers.connectivity_edge_bundle import ConnectivityEdgeBundle
from tvb.adapters.visualizers.covariance import CovarianceVisualizer
from tvb.adapters.visualizers.cross_coherence import CrossCoherenceVisualizer
from tvb.adapters.visualizers.cross_correlation import CrossCorrelationVisualizer
from tvb.adapters.visualizers.eeg_monitor import EegMonitor
from tvb.adapters.visualizers.fourier_spectrum import FourierSpectrumDisplay
from tvb.adapters.visualizers.histogram import HistogramViewer
from tvb.adapters.visualizers.ica import ICA
from tvb.adapters.visualizers.local_connectivity_view import LocalConnectivityViewer
from tvb.adapters.visualizers.matrix_viewer import MappedArrayVisualizer
from tvb.adapters.visualizers.pca import PCA
from tvb.adapters.visualizers.pearson_cross_correlation import PearsonCorrelationCoefficientVisualizer
from tvb.adapters.visualizers.pearson_edge_bundle import PearsonEdgeBundle
from tvb.adapters.visualizers.pse_discrete import DiscretePSEAdapter
from tvb.adapters.visualizers.pse_isocline import IsoclinePSEAdapter
from tvb.adapters.visualizers.region_volume_mapping import ConnectivityMeasureVolumeVisualizer, \
    MappedArrayVolumeVisualizer, MriVolumeVisualizer, RegionVolumeMappingVisualiser
from tvb.adapters.visualizers.sensors import SensorsViewer
from tvb.adapters.visualizers.surface_view import SurfaceViewer, RegionMappingViewer, ConnectivityMeasureOnSurfaceViewer
from tvb.adapters.visualizers.time_series import TimeSeries
from tvb.adapters.visualizers.time_series_volume import TimeSeriesVolumeVisualiser
from tvb.adapters.visualizers.topographic import TopographicViewer
from tvb.adapters.visualizers.tract import TractViewer
from tvb.adapters.visualizers.view_algorithm_category_config import ViewAlgorithmCategoryConfig
from tvb.adapters.visualizers.wavelet_spectrogram import WaveletSpectrogramVisualizer
import tvb.adapters.portlets as portlets_module

from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.fcd import FcdIndex
from tvb.core.entities.model.datatypes.graph import ConnectivityMeasureIndex, CorrelationCoefficientsIndex, \
    CovarianceIndex
from tvb.core.entities.model.datatypes.local_connectivity import LocalConnectivityIndex
from tvb.core.entities.model.datatypes.mode_decompositions import IndependentComponentsIndex, PrincipalComponentsIndex
from tvb.core.entities.model.datatypes.patterns import StimuliRegionIndex, StimuliSurfaceIndex
from tvb.core.entities.model.datatypes.projections import ProjectionMatrixIndex
from tvb.core.entities.model.datatypes.region_mapping import RegionMappingIndex, RegionVolumeMappingIndex
from tvb.core.entities.model.datatypes.sensors import SensorsIndex
from tvb.core.entities.model.datatypes.spectral import CoherenceSpectrumIndex, ComplexCoherenceSpectrumIndex, \
    FourierSpectrumIndex, WaveletCoefficientsIndex
from tvb.core.entities.model.datatypes.structural import StructuralMRIIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.entities.model.datatypes.temporal_correlations import CrossCorrelationIndex
from tvb.core.entities.model.datatypes.time_series import TimeSeriesEEGIndex, TimeSeriesIndex, TimeSeriesMEGIndex, \
    TimeSeriesRegionIndex, TimeSeriesSEEGIndex, TimeSeriesSurfaceIndex, TimeSeriesVolumeIndex
from tvb.core.entities.model.datatypes.tracts import TractsIndex
from tvb.core.entities.model.datatypes.volume import VolumeIndex

from tvb.datatype_removers.remover_connectivity import ConnectivityRemover
from tvb.datatype_removers.remover_region_mapping import RegionMappingRemover, RegionVolumeMappingRemover
from tvb.datatype_removers.remover_sensor import SensorRemover
from tvb.datatype_removers.remover_surface import SurfaceRemover
from tvb.datatype_removers.remover_timeseries import TimeseriesRemover
from tvb.datatype_removers.remover_volume import VolumeRemover

if TvbProfile.current.MATLAB_EXECUTABLE and len(TvbProfile.current.MATLAB_EXECUTABLE) > 0:
    from tvb.adapters.analyzers.bct_adapters import DistanceDBIN, DistanceDWEI, DistanceNETW, DistanceRDA, DistanceRDM, \
        ModularityOCSM, ModularityOpCSMU
    from tvb.adapters.analyzers.bct_centrality_adapters import CentralityEdgeBinary, CentralityEdgeWeighted, \
        CentralityEigenVector, CentralityKCoreness, CentralityKCorenessBD, CentralityNodeBinary, CentralityNodeWeighted, \
        CentralityShortcuts, FlowCoefficients, ParticipationCoefficient, ParticipationCoefficientSign, \
        SubgraphCentrality
    from tvb.adapters.analyzers.bct_clustering_adapters import ClusteringCoefficient, ClusteringCoefficientBU, \
        ClusteringCoefficientWD, ClusteringCoefficientWU, TransitivityBinaryDirected, TransitivityBinaryUnDirected, \
        TransitivityWeightedDirected, TransitivityWeightedUnDirected
    from tvb.adapters.analyzers.bct_degree_adapters import Degree, DegreeIOD, DensityDirected, DensityUndirected, \
        JointDegree, MatchingIndex, Strength, StrengthISOS, StrengthWeights


class IntrospectionRegistry(object):
    """
    This registry gathers classes that have a role in generating DB tables and rows.
    It is used at introspection time, for the following operations:
        - fill-in all rows in the ALGORITHM_CATEGORIES table
        - fill-in all rows in the ALGORITHMS table. Will add BCT algorithms only if Matlab/Octave path is set
        - generate DB tables for all datatype indexes
        - fill-in all rows in the PORTLETS table using data defined in XML files
        - keep an evidence of the datatype index removers
    All classes that subclass AlgorithmCategoryConfig, ABCAdapter, ABCRemover, HasTraitsIndex should be imported here
    and added to the proper dictionary/list.
    e.g. Each new class of type HasTraitsIndex should be imported here and added to the DATATYPES list.
    """
    ADAPTERS = {
        AnalyzeAlgorithmCategoryConfig: [
            CrossCorrelateAdapter,
            PearsonCorrelationCoefficientAdapter,
            FunctionalConnectivityDynamicsAdapter,
            BalloonModelAdapter,
            FourierAdapter,
            ICAAdapter,
            TimeseriesMetricsAdapter,
            NodeCoherenceAdapter,
            NodeComplexCoherenceAdapter,
            NodeCovarianceAdapter,
            PCAAdapter,
            ContinuousWaveletTransformAdapter
        ],
        SimulateAlgorithmCategoryConfig: [SimulatorAdapter],
        UploadAlgorithmCategoryConfig: [
            BRCOImporter,
            CFF_Importer,
            ConnectivityMeasureImporter,
            GIFTISurfaceImporter,
            GIFTITimeSeriesImporter,
            CSVConnectivityImporter,
            MatTimeSeriesImporter,
            EEGMatTimeSeriesImporter,
            NetworkxConnectivityImporter,
            NIFTIImporter,
            ObjSurfaceImporter,
            ProjectionMatrixSurfaceEEGImporter,
            RegionMapping_Importer,
            Sensors_Importer,
            TVBImporter,
            TrackvizTractsImporter,
            ZipTxtTractsImporter,
            ZIPConnectivityImporter,
            ZIPSurfaceImporter
        ],
        ViewAlgorithmCategoryConfig: [
            ConnectivityAnnotationsView,
            BrainViewer,
            DualBrainViewer,
            ImaginaryCoherenceDisplay,
            ConnectivityViewer,
            ConnectivityEdgeBundle,
            CovarianceVisualizer,
            CrossCoherenceVisualizer,
            CrossCorrelationVisualizer,
            EegMonitor,
            FourierSpectrumDisplay,
            HistogramViewer,
            ICA,
            LocalConnectivityViewer,
            MappedArrayVisualizer,
            PCA,
            PearsonCorrelationCoefficientVisualizer,
            PearsonEdgeBundle,
            DiscretePSEAdapter,
            IsoclinePSEAdapter,
            ConnectivityMeasureVolumeVisualizer,
            MappedArrayVolumeVisualizer,
            MriVolumeVisualizer,
            RegionVolumeMappingVisualiser,
            SensorsViewer,
            SurfaceViewer,
            RegionMappingViewer,
            ConnectivityMeasureOnSurfaceViewer,
            TimeSeries,
            TimeSeriesVolumeVisualiser,
            TractViewer,
            TopographicViewer,
            WaveletSpectrogramVisualizer
        ],
        CreateAlgorithmCategoryConfig: [
            AllenConnectomeBuilder,
            ConnectivityCreator,
            LocalConnectivityCreator,
            RegionStimulusCreator,
            SurfaceStimulusCreator
        ],
    }

    if TvbProfile.current.MATLAB_EXECUTABLE and len(TvbProfile.current.MATLAB_EXECUTABLE) > 0:
        BCT_ADAPTERS = [
            DistanceDBIN,
            DistanceDWEI,
            DistanceNETW,
            DistanceRDA,
            DistanceRDM,
            ModularityOCSM,
            ModularityOpCSMU,
            CentralityEdgeBinary,
            CentralityEdgeWeighted,
            CentralityEigenVector,
            CentralityKCoreness,
            CentralityKCorenessBD,
            CentralityNodeBinary,
            CentralityNodeWeighted,
            CentralityShortcuts,
            FlowCoefficients,
            ParticipationCoefficient,
            ParticipationCoefficientSign,
            SubgraphCentrality,
            ClusteringCoefficient,
            ClusteringCoefficientBU,
            ClusteringCoefficientWD,
            ClusteringCoefficientWU,
            TransitivityBinaryDirected,
            TransitivityBinaryUnDirected,
            TransitivityWeightedDirected,
            TransitivityWeightedUnDirected,
            Degree,
            DegreeIOD,
            DensityDirected,
            DensityUndirected,
            JointDegree,
            MatchingIndex,
            Strength,
            StrengthISOS,
            StrengthWeights,
        ]
        ADAPTERS[AnalyzeAlgorithmCategoryConfig].extend(BCT_ADAPTERS)

    DATATYPE_REMOVERS = {
        ConnectivityIndex: ConnectivityRemover,
        RegionMappingIndex: RegionMappingRemover,
        RegionVolumeMappingIndex: RegionVolumeMappingRemover,
        SurfaceIndex: SurfaceRemover,
        SensorsIndex: SensorRemover,
        TimeSeriesIndex: TimeseriesRemover,
        TimeSeriesEEGIndex: TimeseriesRemover,
        TimeSeriesMEGIndex: TimeseriesRemover,
        TimeSeriesSEEGIndex: TimeseriesRemover,
        TimeSeriesRegionIndex: TimeseriesRemover,
        TimeSeriesSurfaceIndex: TimeseriesRemover,
        TimeSeriesVolumeIndex: TimeseriesRemover,
        VolumeIndex: VolumeRemover
    }

    DATATYPES = [ConnectivityIndex, FcdIndex, ConnectivityMeasureIndex, CorrelationCoefficientsIndex, CovarianceIndex,
                 LocalConnectivityIndex, IndependentComponentsIndex, PrincipalComponentsIndex, StimuliRegionIndex,
                 StimuliSurfaceIndex, ProjectionMatrixIndex, RegionMappingIndex, RegionVolumeMappingIndex, SensorsIndex,
                 CoherenceSpectrumIndex, ComplexCoherenceSpectrumIndex, FourierSpectrumIndex, WaveletCoefficientsIndex,
                 StructuralMRIIndex, SurfaceIndex, CrossCorrelationIndex, TimeSeriesEEGIndex, TimeSeriesIndex,
                 TimeSeriesMEGIndex, TimeSeriesRegionIndex, TimeSeriesSEEGIndex, TimeSeriesSurfaceIndex,
                 TimeSeriesVolumeIndex, TractsIndex, VolumeIndex]

    PORTLETS_MODULE = portlets_module

    SIMULATOR_MODULE = SimulatorAdapter.__module__
    SIMULATOR_CLASS = SimulatorAdapter.__name__

    SIMULATION_DATATYPE_MODULE = SimulationState.__module__
    SIMULATION_DATATYPE_CLASS = SimulationState.__name__

    CONNECTIVITY_MODULE = ConnectivityViewer.__module__
    CONNECTIVITY_CLASS = ConnectivityViewer.__name__

    ALLEN_CREATOR_MODULE = AllenConnectomeBuilder.__module__
    ALLEN_CREATOR_CLASS = AllenConnectomeBuilder.__name__

    MEASURE_METRICS_MODULE = TimeseriesMetricsAdapter.__module__
    MEASURE_METRICS_CLASS = TimeseriesMetricsAdapter.__name__

    DISCRETE_PSE_ADAPTER_MODULE = DiscretePSEAdapter.__module__
    DISCRETE_PSE_ADAPTER_CLASS = DiscretePSEAdapter.__name__

    ISOCLINE_PSE_ADAPTER_MODULE = IsoclinePSEAdapter.__module__
    ISOCLINE_PSE_ADAPTER_CLASS = IsoclinePSEAdapter.__name__

    DEFAULT_PORTLETS = {0: {0: 'TimeSeries'}}

    DEFAULT_PROJECT_GID = '2cc58a73-25c1-11e5-a7af-14109fe3bf71'
