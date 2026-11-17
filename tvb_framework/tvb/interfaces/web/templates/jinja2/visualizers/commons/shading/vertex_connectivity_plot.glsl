attribute vec3 aVertexPosition;
attribute vec2 aTextureCoord;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat4 uNMatrix;
uniform bool uDrawLines;
uniform vec3 uLineColor;
uniform float uAlpha;

varying vec4 vColor;
varying vec2 vTextureCoord;
varying float vUseTexture;
uniform float isPicking;
uniform vec3 pickingColor;

void main(void) {
    gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
    vTextureCoord = aTextureCoord;
    vUseTexture = 0.0;
    if (isPicking == 0.0) {
        if (uDrawLines) {
            vColor = vec4(uLineColor, uAlpha);
        } else {
            // The matrix cell colors come from the texture, only the alpha is set here.
            vColor = vec4(0.0, 0.0, 0.0, uAlpha);
            vUseTexture = 1.0;
        }
    } else {
        vColor = vec4(pickingColor, 1.0);
    }
}