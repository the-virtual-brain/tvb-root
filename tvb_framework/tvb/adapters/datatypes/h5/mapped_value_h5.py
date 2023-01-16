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

from tvb.basic.neotraits.api import Attr, HasTraits
from tvb.core.neotraits.h5 import H5File, Scalar


class ValueWrapper(HasTraits):
    data_value = Attr(str)
    data_type = Attr(str)
    data_name = Attr(str)

    @property
    def value(self):
        if "int" == self.data_type.lower():
            return int(self.data_value)
        if "float" == self.data_type.lower():
            return float(self.data_value)
        return self.data_value


class ValueWrapperH5(H5File):

    def __init__(self, path):
        super(ValueWrapperH5, self).__init__(path)
        self.data_value = Scalar(ValueWrapper.data_value, self)
        self.data_type = Scalar(ValueWrapper.data_type, self)
        self.data_name = Scalar(ValueWrapper.data_name, self)
