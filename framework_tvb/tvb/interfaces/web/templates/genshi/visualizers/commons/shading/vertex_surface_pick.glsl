{% include transform.glsl %}

attribute vec3 aVertexPosition;
attribute vec3 aVertexNormal;

attribute vec4 aVertexColor;

varying vec4 vColor;
varying vec3 posInterp;
varying vec3 normInterp;

void main(void) {
    transformed_pos(aVertexPosition, aVertexNormal, gl_Position, posInterp, normInterp);
    vColor = aVertexColor;
}