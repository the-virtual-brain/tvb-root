# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

"""
.. moduleauthor:: Paula Sanz Leon <pau.sleon@gmail.com>
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import tvb.datatypes.time_series as time_series_module
from tvb.basic.neotraits.api import HasTraits, Attr, Int, Float


class BaseTimeseriesMetricAlgorithm(HasTraits):
    """
    This is a base class for all metrics on timeSeries dataTypes.
    Metric means an algorithm computing a single value for an entire TimeSeries.

    """

    time_series = Attr(
        field_type=time_series_module.TimeSeries,
        label="Time Series",
        required=True,
        doc="The TimeSeries for which the metric(s) will be computed.")

    start_point = Float(
        label="Start point (ms)",
        default=500.0,
        required=False,
        doc=""" The start point determines how many points of the TimeSeries will
        be discarded before computing the metric. By default it drops the
        first 500 ms.""")

    segment = Int(
        label="Segmentation factor",
        default=4,
        required=False,
        doc=""" Divide the input time-series into discrete equally sized sequences and
        use the last segment to compute the metric. It is only used when
        the start point is larger than the time-series length.""")

    def evaluate(self):
        """
        This method needs to be implemented in each subclass.
        Will describe current algorithm.

        :return: single numeric value or a dictionary (displayLabel: numeric value) to be persisted.
        """
        raise Exception("Every metric algorithm should implement an 'evaluate' method that returns the metric result.")
