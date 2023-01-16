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

from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Int, trait_property, cached_trait_property, Dim
from tvb.basic.neotraits.ex import TraitValueError


class BazDataType(HasTraits):
    miu = NArray()
    scalar_str = Attr(str, required=False)


class FooDatatype(HasTraits):
    array_float = NArray()
    array_int = NArray(dtype=int, shape=(Dim.any, Dim.any))
    scalar_int = Int()
    abaz = Attr(field_type=BazDataType)
    some_transient = NArray(shape=(Dim.any, Dim.any, Dim.any), required=False)


class BarDatatype(FooDatatype):
    array_str = NArray(dtype='S32', shape=(Dim.any,))


class PropsDataType(HasTraits):
    n_node = Int()

    def __init__(self, **kwargs):
        super(PropsDataType, self).__init__(**kwargs)
        self._weights = None

    @trait_property(NArray(shape=(Dim.any, Dim.any)))
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        if val.shape != (self.n_node, self.n_node):
            raise TraitValueError
        self._weights = val

    @trait_property(Attr(bool))
    def is_directed(self):
        isit = (self.weights == self.weights.T).all()
        # The strict typing is fighting against python conventions
        # numpy.bool_ is not bool ...
        return bool(isit)

    @cached_trait_property(NArray())
    def once(self):
        return self.weights * 22.44
