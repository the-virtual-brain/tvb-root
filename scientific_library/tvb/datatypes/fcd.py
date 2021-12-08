# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Adapter that uses the traits module to generate interfaces for ... Analyzer.

.. moduleauthor:: Francesca Melozzi <france.melozzi@gmail.com>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import tvb.datatypes.time_series as time_series
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List, Float, narray_summary_info



class Fcd(HasTraits):
    array_data = NArray()

    source = Attr(
        field_type=time_series.TimeSeries,
        label="Source time-series",
        doc="Links to the time-series on which FCD is calculated.")

    sw = Float(
        label="Sliding window length (ms)",
        default=120000,
        doc="""Length of the time window used to divided the time series.
                FCD matrix is calculated in the following way: the time series is divided in time window of fixed length and with an overlapping of fixed length.
                The datapoints within each window, centered at time ti, are used to calculate FC(ti) as Pearson correlation.
                The ij element of the FCD matrix is calculated as the Pearson correlation between FC(ti) and FC(tj) arranged in a vector.""")

    sp = Float(
        label="Spanning between two consecutive sliding window (ms)",
        default=2000,
        doc="""Spanning= (time windows length)-(overlapping between two consecutive time window).
                FCD matrix is calculated in the following way: the time series is divided in time window of fixed length and with an overlapping of fixed length.
                The datapoints within each window, centered at time ti, are used to calculate FC(ti) as Pearson correlation.
                The ij element of the FCD matrix is calculated as the Pearson correlation between FC(ti) and FC(tj) arranged in a vector""")

    labels_ordering = List(
        of=str,
        label="Dimension Names",
        default=("Time", "Time", "State Variable", "Mode"),
        doc="""List of strings representing names of each data dimension""")

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {
            "FCD type": self.__class__.__name__,
            "Source": self.source.title,
            "Dimensions": self.labels_ordering
        }
        summary.update(narray_summary_info(self.array_data))
        return summary
