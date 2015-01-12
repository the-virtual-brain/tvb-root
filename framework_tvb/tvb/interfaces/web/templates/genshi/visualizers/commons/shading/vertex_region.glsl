{% include transform.glsl %}
{% include colorscheme.glsl %}

attribute vec3 aVertexPosition;
attribute vec3 aVertexNormal;

attribute vec3 alphaIndices;
attribute vec2 alpha;

uniform vec2 uActivity[${abs(noOfMeasurePoints) + 2} + 127];

varying vec4 vColor;
varying vec3 posInterp;
varying vec3 normInterp;

void main(void) {
    transformed_pos(aVertexPosition, aVertexNormal, gl_Position, posInterp, normInterp);

    vec2 uv = uActivity[int(alphaIndices[0])] * alpha[0] +
              uActivity[int(alphaIndices[1])] * alpha[1] +
              uActivity[int(alphaIndices[2])] * (1.0 - alpha[0] - alpha[1]);

    vColor = colorSchemeLookup(uv);
}
