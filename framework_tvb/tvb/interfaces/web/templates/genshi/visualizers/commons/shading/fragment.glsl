precision mediump float;

{% include 'visualizers/commons/shading/light.glsl' %}

// these are used to draw whole objects in a specific color ignoring vertex attributes.
// for vertex level picking special color buffers are used.
uniform bool uUseVertexColors;
uniform vec4 uMaterialColor;

varying vec4 vColor;
varying vec3 posInterp;
varying vec3 normInterp;

void main(void) {
    if (uUseVertexColors){
        gl_FragColor = light(vColor, posInterp, normInterp);
    } else {
        gl_FragColor = light(uMaterialColor, posInterp, normInterp);
    }
}
