/**
 * Lookup color for activity in a color scheme texture
 */
uniform sampler2D uSampler;
uniform vec2 activityRange;
const float textureSize = 256.0;

vec4 colorSchemeLookup(in vec2 uv){
    //scale activity within given range to [0,1]
    float u;
    u = (uv[0] - activityRange[0] ) / (activityRange[1] - activityRange[0]);
    //scale [0,1] to an interval that will sample the interior of the texture
    u = u * (textureSize - 3.0)/(textureSize - 1.0) + 1.0/(textureSize - 1.0);
    return texture2D(uSampler, vec2(u, uv[1]));
}
