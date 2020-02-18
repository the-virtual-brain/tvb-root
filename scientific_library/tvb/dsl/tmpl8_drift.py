from .base import Model, ModelNumbaDfun
import numexpr
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
"${itemA.name}": numpy.array([${itemA.dimension}])${'' if loop.last else ', \n\t\t\t\t '}\
%endfor
},
        ## doc="""${dynamics.state_variables['V'].exposure}"""
        doc="""state variables"""
        )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=(\
%for i, itemJ in enumerate(exposures):
%if i == 0:
"${itemJ.dimension}"),
        default=("${itemJ.name}", ),
%endif
%endfor),
        doc="The quantities of interest for monitoring for the generic 2D oscillator."
    )

    state_variables = [\
%for itemB in dynamics.state_variables:
'${itemB.name}'${'' if loop.last else ', '}\
%endfor
]

    _nvar = ${dynamics.state_variables.__len__()}
    cvar = numpy.array([0], dtype=numpy.int32)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, ev=numexpr.evaluate):

        % for i, itemC in enumerate(dynamics.state_variables):
        %if (i == 0):
        lc_0 = local_coupling * ${itemC.name}
        %endif
        ${itemC.name} = state_variables[${i},:]
        % endfor

        #[State_variables, nodes]
        c_0 = coupling[0, :]

        %for itemD in const:
        ${itemD.name} = self.${itemD.name}
        %endfor

        derivative = numpy.empty_like(state_variables)

        ## derived variables
        % for i, der_var in enumerate(dynamics.derived_variables):
        ${der_var.name} = ${der_var.value}
        % endfor

        % for i, item in enumerate(dynamics.time_derivatives):
        ##derivative[${i}] = ${item.value}
        ev('${item.value}', out=derivative[${i}])
        % endfor

        return derivative

    def dfun(self, vw, c, local_coupling=0.0):
        lc_0 = local_coupling * vw[0, :, 0]
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_${dfunname}(vw_, c_, \
%for itemE in const:
self.${itemE.name}, \
%endfor
lc_0)

        return deriv.T[..., numpy.newaxis]

## signature is always the number of constants +4. the extras are vw, c_0, lc_0 and dx.
@guvectorize([(float64[:],) * ${const.__len__()+4}], '(n),(m)' + ',()'*${const.__len__()+1} + '->(n)', nopython=True)
def _numba_dfun_${dfunname}(vw, c_0, \
% for itemI in const:
${itemI.name}, \
% endfor
lc_0, dx):
    "Gufunc for ${dfunname} model equations."

    % for i, itemF in enumerate(dynamics.state_variables):
    ${itemF.name} = vw[${i}]
    % endfor

    ## annotate with [0]
    % for itemG in const:
    ${itemG.name} = ${itemG.name}[0]
    % endfor
    c_0 = c_0[0]
    lc_0 = lc_0[0]

    ## derived variables
    % for i, der_var in enumerate(dynamics.derived_variables):
    ${der_var.name} = ${der_var.value}
    % endfor

    % for i, itemH in enumerate(dynamics.time_derivatives):
    dx[${i}] = ${itemH.value}
    % endfor
    \
    \
    ## TVB numpy constant declarations
    <%def name="NArray(nconst)">
    ${nconst.name} = NArray(
        label=":math:`${nconst.name}`",
        default=numpy.array([${nconst.value}]),
        domain=Range(${nconst.dimension}),
        doc="""${nconst.description}"""
    )\
    ##self.${nconst.name} = ${nconst.name}
    </%def>