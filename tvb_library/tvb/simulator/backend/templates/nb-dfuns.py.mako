## -*- coding: utf-8 -*-
##
##
## TheVirtualBrain-Scientific Package. This package holds all simulators, and
## analysers necessary to run brain-simulations. You can use it stand alone or
## in conjunction with TheVirtualBrain-Framework Package. See content of the
## documentation-folder for more details. See also http://www.thevirtualbrain.org
##
## (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
## When using The Virtual Brain for scientific publications, please cite it as explained here:
## https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
##
##

import math
import numpy as np
import numba as nb
sin, cos, exp, log = math.sin, math.cos, math.exp, math.log

<%
    svars = ', '.join(sim.model.state_variables)
    cvars = ', '.join(sim.model.coupling_terms)
    try:
        _dfun_combined = bool(dfun_combined)
    except NameError:
        _dfun_combined = False
%>

## emit physical constants if model provides them (KIonEx-style)
% if hasattr(sim.model, 'dfun_constants'):
% for _cname, _cval in sim.model.dfun_constants.items():
${_cname} = ${_cval}
% endfor
% endif

## emit helper functions if model provides them
% if hasattr(sim.model, 'dfun_helpers'):
% for _fname, _fargs, _fexpr in sim.model.dfun_helpers:
${'' if debug_nojit else '@nb.njit(inline="always")'}
def ${_fname}(${_fargs}):
    return ${_fexpr}

% endfor
% endif

## Strategy A (default): one function per state variable
## Intermediates are recomputed inside each function.
% if not _dfun_combined:
% for svar in sim.model.state_variables:
${'' if debug_nojit else '@nb.njit(inline="always")'}
def dx_${svar}(${svars}, ${cvars}, parmat):
    ## unpack global parameters
% for par in sim.model.global_parameter_names:
    ${par} = ${getattr(sim.model, par)[0]}
% endfor
    ## unpack spatialized parameters
% for par in sim.model.spatial_parameter_names:
    ${par} = parmat[${loop.index}]
% endfor
    pi = np.pi
    ## compute shared intermediates if model provides them
% if hasattr(sim.model, 'dfun_intermediates'):
% for _iname, _iexpr in sim.model.dfun_intermediates:
    ${_iname} = ${_iexpr}
% endfor
% endif
    ## compute dx
    return ${sim.model.state_variable_dfuns[svar]};
% endfor

## Strategy B: single combined function returning a tuple of all derivatives.
## Intermediates computed once; avoids redundant transcendental evaluations.
% else:
${'' if debug_nojit else '@nb.njit(inline="always")'}
def dfun(${svars}, ${cvars}, parmat):
    ## unpack global parameters
% for par in sim.model.global_parameter_names:
    ${par} = ${getattr(sim.model, par)[0]}
% endfor
    ## unpack spatialized parameters
% for par in sim.model.spatial_parameter_names:
    ${par} = parmat[${loop.index}]
% endfor
    pi = np.pi
    ## compute shared intermediates
% if hasattr(sim.model, 'dfun_intermediates'):
% for _iname, _iexpr in sim.model.dfun_intermediates:
    ${_iname} = ${_iexpr}
% endfor
% endif
    ## return tuple of all derivatives
    return (
% for svar in sim.model.state_variables:
        ${sim.model.state_variable_dfuns[svar]}${',' if not loop.last else ''}
% endfor
    )
% endif

