
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
import uuid
from tvb.basic.neotraits.api import Attr
from tvb.core.entities.file.exceptions import MissingDataSetException
from tvb.core.neotraits.h5 import Json, H5File, Scalar, Reference


class OperationGroupH5(H5File):

    def __init__(self, path):
        super(OperationGroupH5, self).__init__(path)
        self.name = Scalar(Attr(str), self, name='name')
        self.range1 = Scalar(Attr(str, required=False), self, name='range1')
        self.range2 = Scalar(Attr(str, required=False), self, name='range2')
        self.range3 = Scalar(Attr(str, required=False), self, name='range3')
        self.gid = Reference(Attr(uuid.UUID), self, name='gid')
        self.fk_launched_in = Scalar(Attr(str), self, name='fk_launched_in')
        self.operation_view_model_gids = Json(Attr(str), self, name='operation_view_model_gids')
        self.is_metric = Scalar(Attr(bool), self, name='is_metric')

    def store(self, operation_group, operation_view_model_gids=None, is_metric=False):
        # type: (OperationGroup) -> None
        self.name.store(operation_group.name)
        self.range1.store(operation_group.range1)
        self.range2.store(operation_group.range2)
        self.range3.store(operation_group.range3)
        self.gid.store(uuid.UUID(operation_group.gid))
        self.fk_launched_in.store(str(operation_group.fk_launched_in))
        self.operation_view_model_gids.store(operation_view_model_gids)
        self.is_metric.store(is_metric)

    def load_into(self, datatype):
        # type: (OperationGroup) -> None
        super(OperationGroupH5, self).load_into(datatype)
        datatype.name = self.name.load()

        try:
            datatype.range1 = self.range1.load()
        except MissingDataSetException:
            datatype.range2 = None

        try:
            datatype.range2 = self.range2.load()
        except MissingDataSetException:
            datatype.range2 = None

        try:
            datatype.range3 = self.range3.load()
        except MissingDataSetException:
            datatype.range3 = None

        datatype.gid = str(self.gid.load())
        datatype.fk_launched_in = self.fk_launched_in.load()
        datatype.is_metric = self.is_metric.load()
        datatype.operation_view_model_gids = self.operation_view_model_gids.load()