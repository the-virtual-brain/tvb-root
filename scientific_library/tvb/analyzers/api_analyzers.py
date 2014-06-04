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
.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>

Define what Analyzers classes will be included in the UI / Online help 
documentation. Python docstring from the classes listed below will be included.

"""

from tvb.analyzers.correlation_coefficient import CorrelationCoefficient
from tvb.analyzers.cross_correlation import CrossCorrelate
from tvb.analyzers.fft import FFT
from tvb.analyzers.fmri_balloon import BalloonModel
from tvb.analyzers.ica import fastICA
from tvb.analyzers.pca import PCA
from tvb.analyzers.node_coherence import NodeCoherence
from tvb.analyzers.node_complex_coherence import NodeComplexCoherence
from tvb.analyzers.node_covariance import NodeCovariance
from tvb.analyzers.metric_kuramoto_index import KuramotoIndex
from tvb.analyzers.metric_variance_global import GlobalVariance
from tvb.analyzers.metric_variance_of_node_variance import VarianceNodeVariance
from tvb.analyzers.wavelet import ContinuousWaveletTransform



### Dictionary {Analyzer Class : Title to appear in final documentation}

ANALYZERS_FOR_DOCUMENTATION = {
    CorrelationCoefficient: "Pearson Correlation Coefficient",
    CrossCorrelate: "Cross-correlation",
    FFT: "Fast Fourier Transform (FFT)",
    BalloonModel: "Model Balloon",
    fastICA: "Independent Component Analysis",
    NodeCoherence: "Node Coherence",
    NodeComplexCoherence: "Node Complex Coherence",
    NodeCovariance: "Node Covariance",
    PCA: "Principal Components Analysis",
    ContinuousWaveletTransform: "Wavelet",
    GlobalVariance: "Global Variance",
    VarianceNodeVariance: "Variance of Node Variance",
    KuramotoIndex: "Kuramoto Index"
}
