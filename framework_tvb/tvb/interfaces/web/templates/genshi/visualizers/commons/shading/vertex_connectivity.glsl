attribute vec3 aVertexPosition;
attribute vec3 aVertexNormal;
attribute vec3 aColor;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat4 uNMatrix;

uniform vec3 uAmbientColor;
uniform vec3 uLightingDirection;
uniform vec3 uDirectionalColor;

uniform bool uUseLighting;
uniform bool drawNodes;
uniform vec3 uColor;
uniform int uColorIndex;
uniform vec3 uColorsArray[7];

uniform float isPicking;
uniform vec3 pickingColor;

varying float picked;
varying vec3 vLightWeighting;
varying vec3 color;

void main(void) {
    gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);

    picked = isPicking;
    if (isPicking == 0.0) {
        if (!uUseLighting) {
            vLightWeighting = vec3(1.0, 1.0, 1.0);
        } else {
            vec4 transformedNormal = uNMatrix * vec4(aVertexNormal, 1.0);
            float directionalLightWeighting = max(dot(transformedNormal.xyz, uLightingDirection), 0.0);
            vLightWeighting = uAmbientColor + uDirectionalColor * directionalLightWeighting;
        }
        if (drawNodes) {
            color = uColor;
        } else {
            if (uColorIndex != -1) {
                color = uColorsArray[uColorIndex];
            } else {
                color = aColor;
            }
        }
    } else {
        color = pickingColor;
        vLightWeighting = vec3(1, 1, 1);
    }
}