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
##  Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
##  Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
##      The Virtual Brain: a simulator of primate brain network dynamics.
##   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
##
##

from tvb.simulator.models.base import Model, ModelNumbaDfun
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class ${dfunname}(ModelNumbaDfun):
    %for mconst in const:
        ${NArray(mconst)}
    %endfor

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={\
%for itemA in dynamics.state_variables:
"${itemA.name}": numpy.array([${itemA.default}])${'' if loop.last else ', \n\t\t\t\t '}\
%endfor
},
        doc="""state variables"""
    )

% if svboundaries:
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={\
%for limit in dynamics.state_variables:
% if (limit.boundaries!='None' and limit.boundaries!=''):
"${limit.name}": numpy.array([${limit.boundaries}])\
% endif
%endfor
},
    )
% endif \

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=(\
%for itemJ in exposures:
%if {loop.first}:
%for choice in (itemJ.choices):
'${choice}', \
%endfor
),
        default=(\
%for defa in (itemJ.default):
'${defa}', \
%endfor
%endif
%endfor
),
        doc="${itemJ.description}"
    )

    state_variables = [\
%for itemB in dynamics.state_variables:
'${itemB.name}'${'' if loop.last else ', '}\
%endfor
]

    _nvar = ${dynamics.state_variables.__len__()}
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        ##lc_0 = local_coupling * vw[0, :, 0]
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_${dfunname}(vw_, c_, \
%for itemE in const:
self.${itemE.name}, \
%endfor
local_coupling)

        return deriv.T[..., numpy.newaxis]

## signature is always the number of constants +4. the extras are vw, c_0, lc_0 and dx.
@guvectorize([(float64[:], float64[:], \
% for i in range(const.__len__()+1):
float64, \
% endfor
float64[:])], '(n),(m)' + ',()'*${const.__len__()+1} + '->(n)', nopython=True)
def _numba_dfun_${dfunname}(vw, coupling, \
% for itemI in const:
${itemI.name}, \
% endfor
local_coupling, dx):
    "Gufunc for ${dfunname} model equations."

    % for i, itemF in enumerate(dynamics.state_variables):
    ${itemF.name} = vw[${i}]
    % endfor

    ## derived variables
    % for der_var in dynamics.derived_variables:
    ${der_var.name} = ${der_var.expression}
    % endfor

    ## conditional variables
    % for con_der in dynamics.conditional_derived_variables:
    if (${con_der.condition}):
        % for case in (con_der.cases):
% if (loop.first):
        ${con_der.name} = ${case}
% elif (loop.last and not loop.first):
    else:
        ${con_der.name} = ${case}
%endif
        % endfor
    % endfor \

    % for j, itemH in enumerate(dynamics.time_derivatives):
    dx[${j}] = ${itemH.expression}
    % endfor
    \
    \
    ## TVB numpy constant declarations
    <%def name="NArray(nconst)">
    ${nconst.name} = NArray(
        label=":math:`${nconst.name}`",
        default=numpy.array([${nconst.default}]),
        % if (nconst.domain != "None" and nconst.domain != ""):
        domain=Range(${nconst.domain}),
        % endif
        doc="""${nconst.description}"""
    )\
    ##self.${nconst.name} = ${nconst.name}
    </%def>
