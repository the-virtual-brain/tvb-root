<%self:cu_kernel_signature name="mpr_dfun">
    unsigned int n_node,
    float * __restrict__ dX,
    float * __restrict__ state,
    float * __restrict__ coupling
</%self:cu_kernel_signature>
{
    const unsigned int id = threadIdx.x;

    ${compile_time_parameters()}

    % if debug:
    printf("id = %d, n_node = %d, blockdim.x = %d\\n", id, n_node, blockDim.x);
    % endif

    <%self:thread_guard limit="n_node">
        % for svar in model.state_variables:
        float ${svar} = ${get2d('state', 'n_node', loop.index, 'id')};
        % endfor

        % for cterm in model.coupling_terms:
        float ${cterm} = ${get2d('coupling', 'n_node', loop.index, 'id')};
        % endfor

        % for svar in model.state_variables:
        dX[${loop.index}*n_node + id] = ${model.state_variable_dfuns[svar]};
        % endfor
    </%self:thread_guard>
}

<%def name="cu_kernel_signature(name)">
__global__ void ${name}(
    ${caller.body()}
    )
</%def>

<%def name="thread_guard(limit)">
    if (threadIdx.x < ${limit})
    {
        ${caller.body()}
    }
</%def>

<%def name="compile_time_parameters()">
% for par in model.parameter_names:
${decl_const_float(par, getattr(model, par)[0])}
% endfor
${decl_const_float('pi', np.pi)}
</%def>

<%def name="get2d(src,n,i,j)">${src}[${i}*${n} + ${j}]</%def>
<%def name="decl_float(name,val)">float ${name} = ${val}f;</%def>
<%def name="decl_const_float(name,val)">const ${decl_float(name,val)}</%def>