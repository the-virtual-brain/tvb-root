/**
 * Do the usual geometric transforms
 */
uniform mat4 uPMatrix;
uniform mat4 uMVMatrix;
uniform mat4 uNMatrix;

void transformed_pos(in vec3 aVertexPosition, in  vec3 aVertexNormal,
        out vec4 position, out vec3 posInterp, out vec3 normInterp ){

    vec4 mvPosition = uMVMatrix * vec4(aVertexPosition, 1.0);
    vec4 transformedNormal = uNMatrix * vec4(aVertexNormal, 1.0);
    posInterp = vec3(mvPosition) / mvPosition.w;
    normInterp = vec3(transformedNormal);
    position = uPMatrix * mvPosition;
}
