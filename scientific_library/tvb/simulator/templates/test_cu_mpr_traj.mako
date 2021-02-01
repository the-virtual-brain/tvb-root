<%namespace name="cu" file="cu-defs.mako" />

<%cu:kernel_signature name="mpr_traj">
    unsigned int n_node,
    float * __restrict__ dX,
    float * __restrict__ state,
    float * __restrict__ coupling,
    float * __restrict__ trace
</%cu:kernel_signature>
{
    const unsigned int id = threadIdx.x;
    float dt = ${dt}f;
    unsigned int nt = ${nt};
    ${cu.compile_time_parameters()}

    % if debug:
    printf("id = %d, n_node = %d, blockdim.x = %d\\n", id, n_node, blockDim.x);
    % endif

    <%cu:thread_guard limit="n_node">

        % for svar in model.state_variables:
        trace[0*2*n_node + ${loop.index}*n_node + id] = state[${loop.index}*n_node + id];
        % endfor

        for (unsigned int t = 0; t < nt; t++)
        {
            % for svar in model.state_variables:
            float ${svar} = ${cu.get2d('state', 'n_node', loop.index, 'id')};
            % endfor

            % for cterm in model.coupling_terms:
            float ${cterm} = ${cu.get2d('coupling', 'n_node', loop.index, 'id')};
            % endfor

            % for svar in model.state_variables:
            dX[${loop.index}*n_node + id] = ${model.state_variable_dfuns[svar]};
            % endfor

            % for svar in model.state_variables:
            state[${loop.index}*n_node + id] += dt * dX[${loop.index}*n_node + id];
            % endfor

            % for svar in model.state_variables:
            trace[(t+1)*2*n_node + ${loop.index}*n_node + id] = state[${loop.index}*n_node + id];
            % endfor
        }
    </%cu:thread_guard>
}