# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
The Mode Decomposition datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""

import numpy
import tvb.datatypes.time_series as time_series
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Int


class PrincipalComponents(HasTraits):
    """
    Result of a Principal Component Analysis (PCA).
    """

    source = Attr(
        field_type=time_series.TimeSeries,
        label="Source time-series",
        doc="Links to the time-series on which the PCA is applied.")

    weights = NArray(
        label="Principal vectors",
        doc="""The vectors of the 'weights' with which each time-series is represented in each component.""")

    fractions = NArray(
        label="Fraction explained",
        doc="""A vector or collection of vectors representing the fraction of
                the variance explained by each principal component.""")

    norm_source = NArray(label="Normalised source time series")

    component_time_series = NArray(label="Component time series")

    normalised_component_time_series = NArray(label="Normalised component time series")

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {
            "Mode decomposition type": self.__class__.__name__,
            "Source": self.source.title
        }
        return summary

    def compute_norm_source(self):
        """Normalised source time-series."""
        self.norm_source = ((self.source.data - self.source.data.mean(axis=0)) /
                            self.source.data.std(axis=0))
        # self.trait["norm_source"].log_debug(owner=self.__class__.__name__)

    # TODO: ??? Any value in making this a TimeSeries datatypes ???
    def compute_component_time_series(self):
        """Compnent time-series."""
        # TODO: Generalise -- it currently assumes 4D TimeSeriesSimulator...
        ts_shape = self.source.data.shape
        component_ts = numpy.zeros(ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.weights[:, :, var, mode]
                ts = self.source.data[:, var, :, mode]
                component_ts[:, var, :, mode] = numpy.dot(w, ts.T).T

        self.component_time_series = component_ts
        # self.trait["component_time_series"].log_debug(owner=self.__class__.__name__)

    # TODO: ??? Any value in making this a TimeSeries datatypes ???
    def compute_normalised_component_time_series(self):
        """normalised_Compnent time-series."""
        # TODO: Generalise -- it currently assumes 4D TimeSeriesSimulator...
        ts_shape = self.source.data.shape
        component_ts = numpy.zeros(ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.weights[:, :, var, mode]
                nts = self.norm_source[:, var, :, mode]
                component_ts[:, var, :, mode] = numpy.dot(w, nts.T).T

        self.normalised_component_time_series = component_ts
        # self.trait["normalised_component_time_series"].log_debug(owner=self.__class__.__name__)


class IndependentComponents(HasTraits):
    """
    Result of an Independent Component Analysis.

    """
    source = Attr(
        field_type=time_series.TimeSeries,
        label="Source time-series",
        doc="Links to the time-series on which the ICA is applied.")

    mixing_matrix = NArray(
        label="Mixing matrix - Spatial Maps",
        required=False,
        doc="""The linear mixing matrix (Mixing matrix) """)

    unmixing_matrix = NArray(
        label="Unmixing matrix - Spatial maps",
        doc="""The estimated unmixing matrix used to obtain the unmixed sources from the data""")

    prewhitening_matrix = NArray(
        label="Pre-whitening matrix",
        doc=""" """)

    n_components = Int(
        label="Number of independent components",
        doc=""" Observed data matrix is considered to be a linear combination
            of :math:`n` non-Gaussian independent components""")

    norm_source = NArray(label="Normalised source time series. Zero centered and whitened.")

    component_time_series = NArray(label="Component time series. Unmixed sources.")

    normalised_component_time_series = NArray(label="Normalised component time series")

    def compute_norm_source(self):
        """Normalised source time-series."""
        self.norm_source = ((self.source.data - self.source.data.mean(axis=0)) /
                            self.source.data.std(axis=0))

    def compute_component_time_series(self):
        ts_shape = self.source.data.shape
        component_ts_shape = (ts_shape[0], ts_shape[1], self.n_components, ts_shape[3])
        component_ts = numpy.zeros(component_ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                ts = self.source.data[:, var, :, mode]
                component_ts[:, var, :, mode] = numpy.dot(w, numpy.dot(k, ts.T)).T
        self.component_time_series = component_ts

    def compute_normalised_component_time_series(self):
        ts_shape = self.source.data.shape
        component_ts_shape = (ts_shape[0], ts_shape[1], self.n_components, ts_shape[3])
        component_nts = numpy.zeros(component_ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                nts = self.norm_source[:, var, :, mode]
                component_nts[:, var, :, mode] = numpy.dot(w, numpy.dot(k, nts.T)).T
        self.normalised_component_time_series = component_nts

    def compute_mixing_matrix(self):
        """
        Compute the linear mixing matrix A, so X = A * S ,
        where X is the observed data and S contain the independent components
            """
        ts_shape = self.source.data.shape
        mixing_matrix_shape = (ts_shape[2], self.n_components, ts_shape[1], ts_shape[3])
        mixing_matrix = numpy.zeros(mixing_matrix_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                temp = numpy.matrix(numpy.dot(w, k))
                mixing_matrix[:, :, var, mode] = numpy.array(numpy.dot(temp.T, (numpy.dot(temp, temp.T)).T))
        self.mixing_matrix = mixing_matrix

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        return {
            "Mode decomposition type": self.__class__.__name__,
            "Source": self.source.title
        }
