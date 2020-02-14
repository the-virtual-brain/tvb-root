from .base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class ${dfunname}(ModelNumbaDfun):

    ##def __init__(self):
    %for mconst in const:
        %if mconst.symbol == 'NArray':
            ${NArray(mconst)}
        %elif mconst.symbol == 'Attr':
            ${Attr(mconst)}
        %elif mconst.symbol == 'Float':
            ${Float(mconst)}
        %elif mconst.symbol == 'Int':
            ${Int(mconst)}
        %endif
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

    state_variables = (\
%for itemB in dynamics.state_variables:
'${itemB.name}'${'' if loop.last else ', '}\
%endfor
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



    _nvar = ${dynamics.state_variables.__len__()}
    cvar = numpy.array([0], dtype=numpy.int32)

        ## % for item in const:
        ## self.${item.name} = ${item.name}
        ## % endfor
        ## self.state_variables = state_variables
        ## self.state_variable_range = state_variable_range

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, ev=numexpr.evaluate):

        % for i, itemC in enumerate(dynamics.state_variables):
        ${itemC.name} = state_variables[${i}, :]
        % if i == 0:
        lc_0 = local_coupling * ${itemC.name}
        % endif
        % endfor

        #[State_variables, nodes]
        c_0 = coupling[0, :]
        ## c_0 = coupling

        # # TODO why does it not default auto to default
        %for itemD in const:
        ${itemD.name} = self.${itemD.name}
        %endfor

        derivative = numpy.empty_like(state_variables)

        % for i, item in enumerate(dynamics.time_derivatives):
        ##derivative[${i}] = ${item.value}
        ev('${item.value}', out=derivative[${i}])
        % endfor

        return derivative

    def dfun(self, vw, c, local_coupling=0.0):
        lc_0 = local_coupling * vw[0, :, 0]
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_g2d(vw_, c_, \
%for itemE in const:
self.${itemE.name}, \
%endfor
lc_0)

        return deriv.T[..., numpy.newaxis]

## signature is always the number of constants +4. the extras are vw, c_0, lc_0 and dx.
@guvectorize([(float64[:],) * ${const.__len__()+4}], '(n),(m)' + ',()'*${const.__len__()+1} + '->(n)', nopython=True)
def _numba_dfun_g2d(vw, c_0, \
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

    % for i, itemH in enumerate(dynamics.time_derivatives):
    dx[${i}] = ${itemH.value}
    % endfor
    \
    \
    ## TVB numpy constant declarations
    <%def name="NArray(nconst)">
    ${nconst.name} = ${nconst.symbol}(
        label=":math:`${nconst.name}`",
        default=numpy.array([${nconst.value}]),
        domain=Range(${nconst.dimension}),
        doc="""${nconst.description}"""
    )\
    ##self.${nconst.name} = ${nconst.name}
    </%def>\
    \
    <%def name="Attr(nconst)">
    ${nconst.name} = ${nconst.symbol}(
        ## todo: adapt fields in LEMS to match TVBs constant requirements more closely
        field_type=${nconst.dimension},
        label=":math:`${nconst.name}`",
        # defaults to super init
        doc = """${nconst.description}"""
    )\
    ##self.${nconst.name} = ${nconst.name}
    </%def>\
    \
    <%def name="Float(nconst)">
    ${nconst.name} = ${nconst.symbol}(
        ## todo: adapt fields in LEMS to match TVBs constant requirements more closely
        label=":math:`${nconst.name}`",
        required=nconst.dimension,
        default=${nconst.value},
        doc = """${nconst.description}"""
    )\
    ##self.${nconst.name} = ${nconst.name}
    </%def>\
    \
    <%def name="Int(nconst)">
    ${nconst.name} = ${nconst.symbol}(
        default=${nconst.value},
        doc = """${nconst.description}"""
    )\
    ##self.${nconst.name} = ${nconst.name}
    </%def>