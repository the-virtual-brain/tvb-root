precision mediump float;

varying vec3 vLightWeighting;
varying vec3 color;
varying float picked;

uniform float uAlpha;

void main(void) {
     if (picked == 0.0) {
            gl_FragColor = vec4(color * vLightWeighting, uAlpha);
     } else {
            gl_FragColor = vec4(color * vLightWeighting, 1.0);
     }
}