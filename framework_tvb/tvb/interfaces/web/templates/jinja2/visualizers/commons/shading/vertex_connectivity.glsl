{% include 'visualizers/commons/shading/transform.glsl' %}
/**
 * This shader draws the connectivity.
 * The cubes and lines are drawn with fixed material colors. Explicit coloring is used for spheres and transparent cortex.
 */
attribute vec3 aVertexPosition;
attribute vec3 aVertexNormal;
attribute vec3 aColor;

uniform float uAlpha;

varying vec4 vColor;
varying vec3 posInterp;
varying vec3 normInterp;

void main(void) {
    transformed_pos(aVertexPosition, aVertexNormal, gl_Position, posInterp, normInterp);
    vColor = vec4(aColor, uAlpha);
}