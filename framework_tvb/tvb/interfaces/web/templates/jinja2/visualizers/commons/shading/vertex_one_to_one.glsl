{% include 'visualizers/commons/shading/transform.glsl' %}
{% include 'visualizers/commons/shading/colorscheme.glsl' %}
/**
 * This shader displays surface level activity.
 */
attribute vec3 aVertexPosition;
attribute vec3 aVertexNormal;

attribute float aActivity;
uniform float uColorScheme;

varying vec4 vColor;
varying vec3 posInterp;
varying vec3 normInterp;

void main(void) {
    transformed_pos(aVertexPosition, aVertexNormal, gl_Position, posInterp, normInterp);
    vec2 uv = vec2(aActivity, uColorScheme);
    vColor = colorSchemeLookup(uv);
}
