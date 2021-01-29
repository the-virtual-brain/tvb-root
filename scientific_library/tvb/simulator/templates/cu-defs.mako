<%def name="kernel_signature(name)">
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