attribute vec3 aVertexPosition;
attribute vec3 aVertexColor;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat4 uNMatrix;
uniform bool uDrawLines;
uniform vec3 uLineColor;
uniform float uAlpha;

varying vec4 vColor;
uniform float isPicking;
uniform vec3 pickingColor;

void main(void) {
    gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
    if (isPicking == 0.0) {
        if (uDrawLines) {
            vColor = vec4(uLineColor, uAlpha);
        } else {
            vColor = vec4(aVertexColor, uAlpha);
        }
    } else {
        vColor = vec4(pickingColor, 1.0);
    }
}