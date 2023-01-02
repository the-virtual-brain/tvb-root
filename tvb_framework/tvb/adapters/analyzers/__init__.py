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

"""
This is the module where all TVB Analyzers are hooked into the framework.

Define in __all__ attribute, modules to be introspected for finding adapters.

"""
ALL_ANALYZERS = ["bct_adapters", "bct_centrality_adapters", "bct_clustering_adapters", "bct_degree_adapters",
                 "cross_correlation_adapter", "fcd_adapter", "fmri_balloon_adapter", "fourier_adapter", "ica_adapter",
                 "metrics_group_timeseries", "node_coherence_adapter", "node_complex_coherence_adapter",
                 "node_covariance_adapter", "pca_adapter", "wavelet_adapter"]
