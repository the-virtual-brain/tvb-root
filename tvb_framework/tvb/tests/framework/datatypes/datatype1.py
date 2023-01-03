# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
from tvb.core.entities.model.model_datatype import DataType
from sqlalchemy import String, Column, Integer, ForeignKey


# TODO review all usages and replace with conftest fixture
class Datatype1(DataType):
    """
    This class is used for testing purposes only.
    """
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)
    row1 = Column(String)
    row2 = Column(String)

    def __init__(self, row1="test", row2="test"):
        super(Datatype1, self).__init__() #You don't need to put self in Python3, but in 2 yes
        self.row1 = row1
        self.row2 = row2

    def __repr__(self):
        formal = "Datatype1(row1=%s, row2=%s)"
        return formal % (self.row1, self.row2)

    def __str__(self):
        informal = "Datatype1(row1, row2)"
        return informal
