/**
 * Lookup color for activity in a color scheme texture
 */
uniform sampler2D uSampler;
uniform vec2 activityRange;
uniform float activityBins;
uniform float centralHoleDiameter;

const float textureSize = 256.0;

vec4 colorSchemeLookup(in vec2 uv){
    //scale activity within given range to [0,1]
    float u;
    u = (uv[0] - activityRange[0] ) / (activityRange[1] - activityRange[0]);
    // treat central values as out of range
    if ( abs(u - 0.5) < centralHoleDiameter / 2.0 ){
        return texture2D(uSampler, vec2(0.0, uv[1]));
    }
    // bin the activity
    u = floor(u * activityBins) / activityBins;
    //scale [0,1] to an interval that will sample the interior of the texture
    u = u * (textureSize - 3.0)/(textureSize - 1.0) + 1.0/(textureSize - 1.0);
    return texture2D(uSampler, vec2(u, uv[1]));
}
