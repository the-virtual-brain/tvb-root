# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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

from tvb.basic.neotraits.api import HasTraits, Attr
from tvb.core.neotraits.h5 import Json, Reference, H5File
from tvb.datatypes.time_series import TimeSeries


class DatatypeMeasure(HasTraits):
    metrics = Attr(dict)

    analyzed_datatype = Attr(
        field_type=TimeSeries,
        label="TimeSeries",
        doc="""Links to the time-series on which the metrics are computed."""
    )


class DatatypeMeasureH5(H5File):

    def __init__(self, path):
        super(DatatypeMeasureH5, self).__init__(path)
        # Actual measure (dictionary Algorithm: single Value)
        self.metrics = Json(DatatypeMeasure.metrics, self)
        # DataType for which the measure was computed.
        self.analyzed_datatype = Reference(DatatypeMeasure.analyzed_datatype, self)
