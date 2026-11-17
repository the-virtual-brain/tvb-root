precision mediump float;

uniform sampler2D uSampler;

varying vec4 vColor;
varying vec2 vTextureCoord;
varying float vUseTexture;

void main(void) {
    if (vUseTexture > 0.5) {
        gl_FragColor = vec4(texture2D(uSampler, vTextureCoord).rgb, vColor.a);
    } else {
        gl_FragColor = vColor;
    }
}