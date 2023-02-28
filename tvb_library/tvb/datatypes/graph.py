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

The Graph datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <paula@tvb.invalid>

"""
import numpy
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List, narray_summary_info
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.time_series import TimeSeries


class Covariance(HasTraits):
    """Covariance datatype."""

    array_data = NArray(dtype=numpy.complex128)

    source = Attr(
        field_type=TimeSeries,
        label="Source time-series",
        doc="Links to the time-series on which NodeCovariance is applied.")

    def summary_info(self):
        summary = {
            "Graph type": self.__class__.__name__,
            "Source": self.source.title
        }
        summary.update(narray_summary_info(self.array_data))
        return summary


class CorrelationCoefficients(HasTraits):
    """Correlation coefficients datatype."""

    # Extreme values for pearson Correlation Coefficients
    PEARSON_MIN = -1
    PEARSON_MAX = 1

    array_data = NArray()

    source = Attr(
        field_type=TimeSeries,
        label="Source time-series",
        doc="Links to the time-series on which Correlation (coefficients) is applied.")

    labels_ordering = List(
        of=str,
        label="Dimension Names",
        default=("Node", "Node", "State Variable", "Mode"),
        doc="""List of strings representing names of each data dimension""")

    def summary_info(self):
        summary = {
            "Graph type": self.__class__.__name__,
            "Source": self.source.title,
            "Dimensions": self.labels_ordering
        }
        summary.update(narray_summary_info(self.array_data))
        return summary


class ConnectivityMeasure(HasTraits):
    """Measurement of based on a connectivity."""

    array_data = NArray()

    connectivity = Attr(field_type=Connectivity)

    def summary_info(self):
        summary = {"Graph type": self.__class__.__name__}
        summary.update(narray_summary_info(self.array_data))
        return summary
