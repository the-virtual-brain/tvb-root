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

The Temporal Correlation datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import tvb.datatypes.time_series as time_series
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List, narray_summary_info


class CrossCorrelation(HasTraits):
    """
    Result of a CrossCorrelation Analysis.
    """
    array_data = NArray()

    source = Attr(
        field_type=time_series.TimeSeries,
        label="Source time-series",
        doc="""Links to the time-series on which the cross_correlation is applied."""
    )

    time = NArray(label="Temporal Offsets", required=False)

    labels_ordering = List(
        of=str,
        label="Dimension Names",
        default=("Offsets", "Node", "Node", "State Variable", "Mode"),
        doc="""List of strings representing names of each data dimension"""
    )

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {
            "Temporal correlation type": self.__class__.__name__,
            "Source": self.source.title,
            "Dimensions": self.labels_ordering
        }
        summary.update(narray_summary_info(self.array_data))
        return summary
