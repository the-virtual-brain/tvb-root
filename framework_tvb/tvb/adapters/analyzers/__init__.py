# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
This is the module where all TVB Analyzers are hooked into the framework.

Define in __all__ attribute, modules to be introspected for finding adapters.
Define in ____xml_folders__ attribute, folders (relative to TVB root) where 
XML interfaces for analyzers are found.
"""
import os

__all__ = ["cross_correlation_adapter", "fmri_balloon_adapter", "fourier_adapter", "ica_adapter",
           "metrics_group_timeseries", "node_coherence_adapter", "node_complex_coherence_adapter", 
           "node_covariance_adapter", "pca_adapter", "wavelet_adapter"]

__xml_folders__ = [os.path.join("adapters", "analyzers", "matlab_interfaces")]

#Import metrics here, so that Traits will find them...
import tvb.analyzers.metric_kuramoto_index
import tvb.analyzers.metric_proxy_metastability
import tvb.analyzers.metric_variance_global
import tvb.analyzers.metric_variance_of_node_variance


