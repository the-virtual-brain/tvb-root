<%namespace name="cu" file="cu-defs.mako" />

<%cu:kernel_signature name="mpr_dfun">
    unsigned int n_node,
    float * __restrict__ dX,
    float * __restrict__ state,
    float * __restrict__ coupling
</%cu:kernel_signature>
{
    const unsigned int id = threadIdx.x;

    ${cu.compile_time_parameters()}

    % if debug:
    printf("id = %d, n_node = %d, blockdim.x = %d\\n", id, n_node, blockDim.x);
    % endif

## TODO compile/default parameters, thread parameters, per node parameters,
## per node per thread for just subset is probably best ROI

    <%cu:thread_guard limit="n_node">
        % for svar in model.state_variables:
        float ${svar} = ${cu.get2d('state', 'n_node', loop.index, 'id')};
        % endfor

        % for cterm in model.coupling_terms:
        float ${cterm} = ${cu.get2d('coupling', 'n_node', loop.index, 'id')};
        % endfor

        % for svar in model.state_variables:
        dX[${loop.index}*n_node + id] = ${model.state_variable_dfuns[svar]};
        % endfor
    </%cu:thread_guard>
}