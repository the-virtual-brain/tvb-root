# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
from tvb.basic.neotraits.api import Attr
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Reference, SparseMatrix, Json
from tvb.datatypes.equations import Equation
from tvb.datatypes.local_connectivity import LocalConnectivity


class LocalConnectivityH5(H5File):
    def __init__(self, path):
        super(LocalConnectivityH5, self).__init__(path)
        self.surface = Reference(LocalConnectivity.surface, self)
        # this multidataset accessor works but something is off about it
        # this would be clearer
        # self.matrix, self.matrixindices, self.matrixindptr
        self.matrix = SparseMatrix(LocalConnectivity.matrix, self)
        # equation is an inlined reference
        # should this be a special equation scalar field?
        # or this?
        # this is clear about the structure, but obviously breaks the default store/load
        # self.equation_equation = Scalar(Equation.equation, self)
        # self.equation_parameters = Scalar(Equation.parameters, self)

        self.equation = Scalar(Attr(str), self, name='equation')
        self.cutoff = Scalar(LocalConnectivity.cutoff, self)

    # equations are such a special case that we will have to implement custom load store

    def store(self, datatype, scalars_only=False):
        self.surface.store(datatype.surface)
        self.matrix.store(datatype.matrix)
        self.cutoff.store(datatype.cutoff)
        self.equation.store(datatype.equation.to_json(datatype.equation))

    def load_into(self, datatype):
        datatype.gid = self.gid.load()
        datatype.matrix = self.matrix.load()
        datatype.cutoff = self.cutoff.load()
        eq = self.equation.load()
        eq = datatype.equation.from_json(eq)
        datatype.equation = eq

    def get_min_max_values(self):
        metadata = self.matrix.get_metadata()
        return metadata.min, metadata.max
