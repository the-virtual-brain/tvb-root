# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
The forms editing one Projection of the Hybrid Simulator.

The coupling functions are the ones of ``tvb.simulator.hybrid.coupling``, not the classic Simulator's:
their ``pre`` and ``post`` hooks are called differently, and some of them carry parameters the classic
ones do not (``e0``, ``v0`` and ``use_classic`` on SigmoidalJansenRit, ``dynamic`` and ``globalT`` on
PreSigmoidal). Their forms are therefore built from each class's own traits rather than borrowed from
``coupling_forms``.
"""

from tvb.basic.neotraits.api import Attr, EnumAttr, Float, Int, List, NArray
from tvb.core.entities.file.simulator.view_model import HybridCouplingFunctionsEnum
from tvb.core.neotraits.forms import ArrayField, BoolField, Form, FloatField, IntField, MultiSelectField, \
    Option, SelectField


class HybridCouplingForm(Form):
    """
    The parameters of one hybrid coupling function, one field per trait the class declares.
    """

    def __init__(self, coupling_class):
        super(HybridCouplingForm, self).__init__()
        self.coupling_class = coupling_class
        for name in coupling_class.declarative_attrs:
            if name.startswith('_') or name == 'gid':
                continue
            field = self._field_for(getattr(coupling_class, name))
            if field is not None:
                setattr(self, name, field)

    @staticmethod
    def _field_for(attribute):
        if isinstance(attribute, NArray):
            return ArrayField(attribute)
        if isinstance(attribute, Int):
            return IntField(attribute)
        if isinstance(attribute, Float):
            return FloatField(attribute)
        if isinstance(attribute, Attr) and attribute.field_type is bool:
            return BoolField(attribute)
        return None


class NamesMultiSelectField(MultiSelectField):
    """
    A multi select offering names as they are: the default one title-cases its labels, which would show
    the Jansen-Rit variable y1 as Y1.
    """

    def options(self):
        for i, choice in enumerate(self.trait_attribute.element_choices):
            yield Option(
                id='{}_{}'.format(self.name, i),
                value=choice,
                label=str(choice),
                checked=self.data is not None and choice in self.data
            )


class HybridProjectionForm(Form):
    """
    What is edited on one Projection besides its coupling function's parameters: the coupling variables
    it reads and writes, its scale, the conduction speed turning lengths into delays, and the coupling
    function itself.

    The coupling variables offered are the source and target Models' own, so the form is built for a
    given pair of Models. Its ad hoc Lists carry no field_name, so the controller fills them and reads
    them itself.
    """

    def __init__(self, source_names, target_names):
        super(HybridProjectionForm, self).__init__()
        self.source_cvar = NamesMultiSelectField(
            List(of=str, label='Source coupling variables', choices=tuple(source_names),
                 doc='The coupling variables of the source Model this Projection reads'),
            name='source_cvar')
        self.target_cvar = NamesMultiSelectField(
            List(of=str, label='Target coupling variables', choices=tuple(target_names),
                 doc='The coupling variables of the target Model this Projection writes into'),
            name='target_cvar')
        self.scale = FloatField(Float(label='Scale', default=1.0, required=True,
                                      doc='Factor applied to the weighted sum, after the coupling function'),
                                name='scale')
        self.cv = FloatField(Float(label='Conduction speed (mm/ms)', default=3.0, required=True,
                                   doc='Conduction speed turning the Connectivity tract lengths into delays'),
                             name='cv')
        self.coupling = SelectField(
            EnumAttr(label='Coupling function', default=HybridCouplingFunctionsEnum.LINEAR,
                     doc='The function applied around the weighted sum of the delayed source activity'),
            name='coupling')

    def fill_from_projection(self, projection):
        self.source_cvar.data = list(projection.source_cvar)
        self.target_cvar.data = list(projection.target_cvar)
        self.scale.data = projection.scale
        self.cv.data = projection.cv
        self.coupling.data = type(projection.coupling)
