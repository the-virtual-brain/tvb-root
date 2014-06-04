# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

Define what DataType classes are to be included in the Online-Help documentation.
Python doc from the classes listed bellow will be included.

"""

from tvb.datatypes.arrays_data import MappedArrayData
from tvb.datatypes.connectivity_data import ConnectivityData
from tvb.datatypes.graph_data import ConnectivityMeasureData, CovarianceData
from tvb.datatypes.mapped_values import DatatypeMeasure, ValueWrapper
from tvb.datatypes.mode_decompositions_data import IndependentComponentsData, PrincipalComponentsData
from tvb.datatypes.patterns_data import StimuliRegionData, StimuliSurfaceData
from tvb.datatypes.projections_data import ProjectionRegionEEGData, ProjectionSurfaceEEGData
from tvb.datatypes.simulation_state import SimulationState
from tvb.datatypes.temporal_correlations_data import CrossCorrelationData
from tvb.datatypes.volumes_data import VolumeData
import tvb.datatypes.sensors_data as sensors
import tvb.datatypes.spectral_data as spectral
import tvb.datatypes.surfaces_data as surfaces 
import tvb.datatypes.time_series_data as timeseries
import tvb.datatypes.lookup_tables as lookup_tables
    

### Dictionary {DataType Class : Title to appear in final documentation}
### We need to define this dictionary, because not all DataTypea are ready to be exposed in documentation.

DATATYPES_FOR_DOCUMENTATION = {

    ## Category 2: Raw Entities - GREEN colored
    ConnectivityData: "Connectivity",
    VolumeData: "Volume",
    surfaces.BrainSkullData: "Brain Skull",
    surfaces.CortexData: "Cortex",
    surfaces.CorticalSurfaceData: "Cortical Surface",
    surfaces.SurfaceData: "Surface",
    surfaces.SkinAirData: "Skin Air",
    surfaces.SkullSkinData: "Skull Skin",

    ## Category 3: Adjacent Entities (Raw or pre-computed by Creators) - YELLOW color
    sensors.SensorsEEGData: "Sensors EEG",
    sensors.SensorsMEGData: "Sensors MEG",
    sensors.SensorsInternalData: "Sensors Internal",
    StimuliRegionData: "Stimuli Region",
    StimuliSurfaceData: "Stimuli Surface",
    ProjectionRegionEEGData: "Projection Region to EEG",
    ProjectionSurfaceEEGData: "Projection Surface to EEG",
    surfaces.LocalConnectivityData: "Local Connectivity",
    surfaces.RegionMappingData: "Region Mapping",
    lookup_tables.PsiTable: "Psi Lookup Table",
    lookup_tables.NerfTable: "Nerf Lookup Table",
    SimulationState: "Simulation State",

    ## Category 1: Array-like entities (mainly computed by Simulator) - RED color
    timeseries.TimeSeriesData: "Time Series",
    timeseries.TimeSeriesEEGData: "Time Series EEG",
    timeseries.TimeSeriesMEGData: "Time Series MEG",
    timeseries.TimeSeriesRegionData: "Time Series Region",
    timeseries.TimeSeriesSurfaceData: "Time Series Surface",
    timeseries.TimeSeriesVolumeData: "Time Series Volume",
    ValueWrapper: "Single Value",
    MappedArrayData: "MappedArray",
    DatatypeMeasure: "DataType Measure",
    ConnectivityMeasureData: "Connectivity Measure",

    ## Category 4: Analyzers Results - BLUE color
    CrossCorrelationData: "Cross Correlation",
    CovarianceData: "Covariance",
    spectral.CoherenceSpectrumData: "Coherence Spectrum",
    spectral.FourierSpectrumData: "Fourier Spectrum",
    spectral.WaveletCoefficientsData: "Wavelet Coefficient",
    IndependentComponentsData: "Independent Components",
    PrincipalComponentsData: "Principal Components"
}


