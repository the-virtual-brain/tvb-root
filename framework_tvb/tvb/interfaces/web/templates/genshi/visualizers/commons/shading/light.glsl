/**
 * Implements phong lighting with a directional and a point light
 */
uniform vec3 uAmbientColor;
uniform vec3 uLightingDirection;
uniform vec3 uDirectionalColor;

uniform float uMaterialShininess;
uniform vec3 uPointLightingLocation;
uniform vec3 uPointLightingSpecularColor;

vec4 light(in vec4 materialColor, in vec3 posInterp, in vec3 normInterp){
    vec3 normal = normalize(normInterp);
    vec3 lightDirection = normalize(uPointLightingLocation - posInterp);
    vec3 reflectionDirection = reflect(-lightDirection, normal);
    vec3 eyeDirection = normalize(-posInterp);

    float directionalLightWeighting = max(dot(normal, uLightingDirection), 0.0);
    float specularLightWeighting = pow(max(dot(reflectionDirection, eyeDirection), 0.0), uMaterialShininess);

    vec3 diffuse = uAmbientColor + uDirectionalColor * directionalLightWeighting;
    vec3 light = materialColor.rgb * diffuse + uPointLightingSpecularColor * specularLightWeighting;
    return vec4(light, materialColor.a);
}
