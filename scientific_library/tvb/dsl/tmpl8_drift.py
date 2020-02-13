from .base import Model, ModelNumbaDfun
import numexpr
import numpy
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
%for item in dynamics.state_variables:
"${item.name}": numpy.array([${item.dimension}])${'' if loop.last else ', \n\t\t\t\t '}\
%endfor
},
        doc="""${dynamics.state_variables['V'].exposure}"""
        )

    state_variables = (\
%for item in dynamics.state_variables:
'${item.name}'${'' if loop.last else ', '}\
%endfor
)

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("V", "W", "V + W", "V - W"),
        default=("V",),
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

        % for i, item in enumerate(dynamics.state_variables):
        ${item.name} = state_variables[${i}, :]
        % endfor

        #[State_variables, nodes]
        c_0 = coupling[0, :]
        ## c_0 = coupling

        # # TODO why does it not default auto to default
        %for item in const:
        ${item.name} = self.${item.name}
        %endfor

        lc_0 = local_coupling * V
        derivative = numpy.empty_like(state_variables)

        # TODO fixed the acceptance of ** but it will process *** now as well. However not as an operand but as a value or node
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
%for item in const:
self.${item.name}, \
%endfor
lc_0)

        return deriv.T[..., numpy.newaxis]



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