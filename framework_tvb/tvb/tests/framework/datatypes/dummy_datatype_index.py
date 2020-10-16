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

from sqlalchemy import Column, Integer, String, ForeignKey
from tvb.core.entities.model.model_datatype import DataType
from tvb.tests.framework.datatypes.dummy_datatype import DummyDataType


class DummyDataTypeIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    row1 = Column(String)
    row2 = Column(String)

    def fill_from_has_traits(self, datatype):
        # type: (DummyDataType) -> None
        super(DummyDataTypeIndex, self).fill_from_has_traits(datatype)
        self.row1 = datatype.row1
        self.row2 = datatype.row2

    @staticmethod
    def accepted_filters():
        filters = DataType.accepted_filters()
        return filters
