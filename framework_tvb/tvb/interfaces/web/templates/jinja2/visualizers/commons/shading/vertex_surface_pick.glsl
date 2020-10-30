{% include 'visualizers/commons/shading/transform.glsl' %}
{% include 'visualizers/commons/shading/colorscheme.glsl' %}
/** This shader combines two behaviours
 * It can display surface activity using a color pallete like vertex_one_to_one.
 * It can also color vertices using a explicit color buffer. This is used for vertex picking.
 * The uniform uUseActivity selects the behaviour.
 */
attribute vec3 aVertexPosition;
attribute vec3 aVertexNormal;

attribute float aActivity;
uniform float uColorScheme;

uniform bool uUseActivity;
attribute vec4 aVertexColor;

varying vec4 vColor;
varying vec3 posInterp;
varying vec3 normInterp;

void main(void) {
    transformed_pos(aVertexPosition, aVertexNormal, gl_Position, posInterp, normInterp);

    if (uUseActivity){
        vec2 uv = vec2(aActivity, uColorScheme);
        vColor = colorSchemeLookup(uv);
    }else{
        vColor = aVertexColor;
    }
}