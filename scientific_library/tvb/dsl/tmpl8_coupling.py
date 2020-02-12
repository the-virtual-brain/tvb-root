from tvb.basic.neotraits.api import NArray, Final, List, Range, HasTraits
import numpy

# Coupling class
class ${couplingname}(HasTraits):

    r"""
    Provides a difference coupling function, between pre and post synaptic
    activity of the form
    .. math::
        a G_ij (x_j - x_i)
    """

    def __init__(self):
    %for cconst in couplingconst:
        %if cconst.symbol == 'NArray':
            ${NArray(cconst)}\
        %elif cconst.symbol == 'Attr':
            ${Attr(cconst)}\
        %elif cconst.symbol == 'Float':
            ${Float(cconst)} \
        %elif cconst.symbol == 'Int':
            ${Int(cconst)} \
        %endif
     %endfor

    def __str__(self):
        return simple_gen_astr(self, \
%for item in couplingconst:
${item.name})
% endfor

    %for item in couplingfunctions:
    def ${item.name}(self, \
        % if item.name == 'pre':
            %for param in couplingparams:
${param.name}${'' if loop.last else ', '}\
            % endfor
):
        return ${item.value}

        % elif item.name == 'post':
            % for reqs in couplingreqs:
${reqs.name}${'' if loop.last else ', '}\
            % endfor
):
        %for cnst in couplingconst:
        ${cnst.name} = self.${cnst.name}.default
        % endfor
        return ${item.value}

        % endif
    % endfor

        ## TVB numpy constant declarations
        <%def name="NArray(nconst)">
        ${nconst.name} = ${nconst.symbol}(
            label=":math:`${nconst.name}`",
            default=numpy.array([${nconst.value}]),
            domain = Range(${nconst.dimension}),
            doc = """${nconst.description}"""
        )
        self.${nconst.name} = ${nconst.name}
        </%def>\
        \
        <%def name="Attr(nconst)">
        ${nconst.name} = ${nconst.symbol}(
            ## todo: adapt fields in LEMS to match TVBs constant requirements more closely
            field_type=${nconst.dimension},
            label=":math:`${nconst.name}`",
            # defaults to super init
            doc = """${nconst.description}"""
        )
        self.${nconst.name} = ${nconst.name}
        </%def>\
        \
        <%def name="Float(nconst)">
        ${nconst.name} = ${nconst.symbol}(
            ## todo: adapt fields in LEMS to match TVBs constant requirements more closely
            label=":math:`${nconst.name}`",
            required=nconst.dimension,
            default=${nconst.value},
            doc = """${nconst.description}"""
        )
        self.${nconst.name} = ${nconst.name}
        </%def>\
        \
        <%def name="Int(nconst)">
        ${nconst.name} = ${nconst.symbol}(
            default=${nconst.value},
            doc = """${nconst.description}"""
        )
        self.${nconst.name} = ${nconst.name}
        </%def>