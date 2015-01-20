{% include transform.glsl %}
{% include colorscheme.glsl %}
/**
 * This shader displays region level activity. The activity is stored in the uniform array.
 * aVertexRegion is the mapping from vertices to region indices.
 */
attribute vec3 aVertexPosition;
attribute vec3 aVertexNormal;

attribute float aVertexRegion;
// 127 is the legend granularity
uniform vec2 uActivity[${abs(noOfMeasurePoints) + 2} + 127];

varying vec4 vColor;
varying vec3 posInterp;
varying vec3 normInterp;

void main(void) {
    transformed_pos(aVertexPosition, aVertexNormal, gl_Position, posInterp, normInterp);

    vec2 uv = uActivity[int(aVertexRegion)];

    vColor = colorSchemeLookup(uv);
}
