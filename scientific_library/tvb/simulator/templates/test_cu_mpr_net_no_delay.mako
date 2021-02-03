<%inherit file="cu-base.mako"/>
<%namespace name="cu" file="cu-defs.mako" />

<%block name="kernel_args">
    unsigned int n_node,
    float * __restrict__ dX,
    float * __restrict__ state,
    float * __restrict__ weights,
    float * __restrict__ trace
</%block>

<%block name="kernel_setup">
    const unsigned int id = threadIdx.x;
    float dt = ${dt}f;
    unsigned int nt = ${nt};
    ${cu.compile_time_parameters()}

    % if debug:
    printf("id = %d, n_node = %d, blockdim.x = %d\\n", id, n_node, blockDim.x);
    % endif
</%block>

<%cu:thread_guard limit="n_node">
    <%cu:time_loop var="t" stop="nt">
        ${cu.loop_unpack_states()}
        ${cu.loop_compute_coupling_terms()}
        ${cu.loop_compute_derivatives(out='dX')}
        ${cu.loop_euler_update()}
        ${cu.loop_update_trace()}
    </%cu:time_loop>
</%cu:thread_guard>

## is this going at it the wrong way?
## a template should 