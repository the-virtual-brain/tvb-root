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

"""
module docstring
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

from tvb.basic.traits.itree_model import TypeNode, SelectTypeNode, DictNode, EnumerateNode
from tvb.simulator.simulator import Simulator
from pprint import pprint


class TestExperimentalInputTree():

    def test_experimental_does_not_crash(self):
        sim2 = Simulator()
        sim2.trait.bound = 'attributes-only'
        itree_exp = sim2.interface_experimental
        # pprint(itree_exp)


    def _cmp_attributes_or_options(self, itree_exp, itree):
        '''
        This compares the tree structures. It tests that the trees have the similar topology.
        The only attribute tested is the node name
        '''
        assert len(itree_exp) == len(itree)

        for nexp, ndict in zip(itree_exp, itree):
            assert nexp.name == ndict.get('name')

            if isinstance(nexp, SelectTypeNode):
                self._cmp_attributes_or_options(nexp.options, ndict['options'])
            elif isinstance(nexp, TypeNode):
                self._cmp_attributes_or_options(nexp.attributes, ndict['attributes'])
            elif isinstance(nexp, DictNode):
                self._cmp_attributes_or_options(nexp.attributes, ndict['attributes'])
            elif isinstance(nexp, EnumerateNode):
                self._cmp_attributes_or_options(nexp.options, ndict['options'])


    def test_experimental_similar_to_default(self):
        sim = Simulator()
        sim.trait.bound = 'attributes-only'
        itree = sim.interface['attributes']

        sim2 = Simulator()
        sim2.trait.bound = 'attributes-only'
        itree_exp = sim2.interface_experimental

        self._cmp_attributes_or_options(itree_exp, itree)

