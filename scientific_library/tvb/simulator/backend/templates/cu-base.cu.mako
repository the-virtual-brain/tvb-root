<%namespace name="cu" file="cu-defs.mako" />

__global__ void ${kernel_name}(
<%block name="kernel_args"/>
)
{
<%block name="kernel_setup"/>

${self.body()}
}