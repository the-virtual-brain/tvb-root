## -*- coding: utf-8 -*-
##
##
## TheVirtualBrain-Scientific Package. This package holds all simulators, and
## analysers necessary to run brain-simulations. You can use it stand alone or
## in conjunction with TheVirtualBrain-Framework Package. See content of the
## documentation-folder for more details. See also http://www.thevirtualbrain.org
##
## (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
##
## This program is free software: you can redistribute it and/or modify it under the
## terms of the GNU General Public License as published by the Free Software Foundation,
## either version 3 of the License, or (at your option) any later version.
## This program is distributed in the hope that it will be useful, but WITHOUT ANY
## WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
## PARTICULAR PURPOSE.  See the GNU General Public License for more details.
## You should have received a copy of the GNU General Public License along with this
## program.  If not, see <http://www.gnu.org/licenses/>.
##
##
##   CITATION:
## When using The Virtual Brain for scientific publications, please cite it as follows:
##
##   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
##   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
##       The Virtual Brain: a simulator of primate brain network dynamics.
##   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
##
##

def dfuns(dX, state, cX, parmat, **mparams):
% for par in sim.model.global_parameter_names:
    ${par} = ${getattr(sim.model, par)[0]}
% endfor

% for par in sim.model.spatial_parameter_names:
    ${par} = parmat[${loop.index}]
% endfor

    pi = np.pi

% for _, cterm in zip(sim.model.cvar, sim.model.coupling_terms):
    ${cterm} = cX[${loop.index}]
% endfor

% for svar in sim.model.state_variables:
    ${svar} = state[${loop.index}]
% endfor

    # compute dfuns
% for svar in sim.model.state_variables:
    dX = tt.set_subtensor(dX[${loop.index}], ${sim.model.state_variable_dfuns[svar]});
% endfor

    return dX
