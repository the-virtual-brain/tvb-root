# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

"""
A collection of neuronal dynamics models.

Specific models inherit from the abstract class Model.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Gaurav Malhotra <Gaurav@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

_module_models = {
    'base': 'Model'.split(', '),
    'epileptor': 'Epileptor, Epileptor2D'.split(', '),
    'epileptor_rs': 'EpileptorRestingState'.split(', '),
    'epileptorcodim3': 'EpileptorCodim3, EpileptorCodim3SlowMod'.split(', '),
    'hopfield': 'Hopfield'.split(', '),
    'jansen_rit': 'JansenRit, ZetterbergJansen'.split(', '),
    'larter_breakspear': 'LarterBreakspear'.split(', '),
    'linear': 'Linear'.split(', '),
    'oscillator': 'Generic2dOscillator, Kuramoto, SupHopf'.split(', '),
    'stefanescu_jirsa': 'ReducedSetFitzHughNagumo, ReducedSetHindmarshRose'.split(', '),
    'wilson_cowan': 'WilsonCowan'.split(', '),
    'wong_wang': 'ReducedWongWang'.split(', '),
    'wong_wang_exc_inh': 'ReducedWongWangExcInh'.split(', '),
    'zerlaut': 'ZerlautFirstOrder, ZerlautSecondOrder'.split(', '),
}