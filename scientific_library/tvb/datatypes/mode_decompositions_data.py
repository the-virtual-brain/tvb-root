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
The Data component of Spectral datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays
import tvb.datatypes.time_series as time_series
from tvb.basic.traits.types_mapped import MappedType



class PrincipalComponentsData(MappedType):
    """
    Result of a Principal Component Analysis (PCA).
    """

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which the PCA is applied.")

    weights = arrays.FloatArray(
        label="Principal vectors",
        doc="""The vectors of the 'weights' with which each time-series is
            represented in each component.""",
        file_storage=core.FILE_STORAGE_EXPAND)

    fractions = arrays.FloatArray(
        label="Fraction explained",
        doc="""A vector or collection of vectors representing the fraction of
            the variance explained by each principal component.""",
        file_storage=core.FILE_STORAGE_EXPAND)

    norm_source = arrays.FloatArray(
        label="Normalised source time series",
        file_storage=core.FILE_STORAGE_EXPAND)

    component_time_series = arrays.FloatArray(
        label="Component time series",
        file_storage=core.FILE_STORAGE_EXPAND)

    normalised_component_time_series = arrays.FloatArray(
        label="Normalised component time series",
        file_storage=core.FILE_STORAGE_EXPAND)

    __generate_table__ = True



class IndependentComponentsData(MappedType):
    """
    Result of TEMPORAL (Fast) Independent Component Analysis
    """

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which the ICA is applied.")

    mixing_matrix = arrays.FloatArray(
        label="Mixing matrix - Spatial Maps",
        doc="""The linear mixing matrix (Mixing matrix) """)

    unmixing_matrix = arrays.FloatArray(
        label="Unmixing matrix - Spatial maps",
        doc="""The estimated unmixing matrix used to obtain the unmixed
            sources from the data""")

    prewhitening_matrix = arrays.FloatArray(
        label="Pre-whitening matrix",
        doc=""" """)

    n_components = basic.Integer(
        label="Number of independent components",
        doc=""" Observed data matrix is considered to be a linear combination
        of :math:`n` non-Gaussian independent components""")

    norm_source = arrays.FloatArray(
        label="Normalised source time series. Zero centered and whitened.",
        file_storage=core.FILE_STORAGE_EXPAND)

    component_time_series = arrays.FloatArray(
        label="Component time series. Unmixed sources.",
        file_storage=core.FILE_STORAGE_EXPAND)

    normalised_component_time_series = arrays.FloatArray(
        label="Normalised component time series",
        file_storage=core.FILE_STORAGE_EXPAND)

    __generate_table__ = True

