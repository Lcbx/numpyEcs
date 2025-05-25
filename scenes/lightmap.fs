#version 330

in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

in vec4 fragShadowClipSpace;

uniform sampler2D texture0;            // diffuse

uniform vec3 lightDir;
uniform sampler2D shadowDepthMap;      // classic depth map (R channel)
uniform sampler2D shadowPenumbraMap;   // RGB: [meshID, distX, distY]

out vec4 finalColor;

bool between(vec2 v, vec2 bottomLeft, vec2 topRight) {                                               
	vec2 s = step(bottomLeft, v) - step(topRight, v);                                                
	return bool(s.x * s.y);
}

float random(vec2 co) {                                                                              
	return fract(dot(co, vec2(3,8)) * dot(co.yx, vec2(7,5)) * 0.03);
}

vec2 get_dir(vec2 encoded){
    return encoded * 2.0 - 1.0;
}

const int OFFSETS_LEN = 8;
const vec2 OFFSETS[OFFSETS_LEN] = vec2[OFFSETS_LEN](
    vec2( 1,  0),   vec2(-1,  0),
    vec2( 0,  1),   vec2( 0, -1)
    ,vec2( 1,  1), vec2(-1,  1),
     vec2( 1, -1), vec2(-1, -1)
);

void main() {
    // basic lambert + texture
    vec3 albedo = fragColor.rgb * texture(texture0, fragTexCoord).rgb;

    // project into shadow‐map UV
    vec3 proj = fragShadowClipSpace.xyz / fragShadowClipSpace.w;
    proj = proj*0.5 + 0.5;
    vec2 shadowUv = proj.xy;
    if (!between(shadowUv, vec2(0.0), vec2(1.0))) {
        finalColor = vec4(albedo, 1.0);
        return;
    }

    // fetch depth and meshID+distance
    float occluderDepth = texture(shadowDepthMap, shadowUv).r;

    // NOTE: industry standard seems to be to apply the bias when creating the shadow map
    // -> not when sampling it  
    // bias to avoid self‐shadow acne
    float NDotL = dot(fragNormal, lightDir);
    float bias  = 0.001 * (1 + NDotL);

    // determine occlusion
    bool occluded = proj.z > occluderDepth + bias;

    finalColor = vec4(albedo, 1.0);

    float shadowFactor = 1;
    if (occluded) {
        shadowFactor = 0.5;
        //shadowFactor = occluderDepth;
    }
    else
        //if( abs(NDotL) > 0.1  )
    {
        vec3  penumbra = texture(shadowPenumbraMap, shadowUv).rgb;
        vec2 penDir = get_dir(penumbra.gb);
        float distToEdgeSq = dot(penDir, penDir);
        float f = distToEdgeSq;
        
        // multiple by shadowmap size
        // TODO : passs it as uniform
        f *= 1024;

        if(f>1) return;

        float occluderDepth = texture(shadowDepthMap, shadowUv + penDir).r;
        //vec2 window_size = vec2(800, 500);
        //for (int i = 0; i < OFFSETS_LEN; ++i) { 
        //    float depth = texture(shadowDepthMap, shadowUv + penDir + OFFSETS[i]/window_size).r;
        //    if(depth < occluderDepth) occluderDepth = depth;
        //}
        float fragmentDepth = proj.z;
        float occlusionDistance = fragmentDepth - (occluderDepth); // + bias);
        if(occlusionDistance < 0){
            return;
        }
        occlusionDistance *= 10;
        occlusionDistance = smoothstep(0,1,occlusionDistance);

        //float noise = random(gl_FragCoord.xy);
        //float noiseStrength = 0.3;
        //if(f < 0.99) f *= (1.0 + noise * noiseStrength - noiseStrength);

        f = smoothstep(0, 1, f);
        f = mix(0.5, 1, f);
        shadowFactor = f;
        shadowFactor = occlusionDistance;
    }
    //finalColor = vec4(fragNormal, 1.0);
    //finalColor = vec4(lightDir, 1.0);
    finalColor.rgb *= shadowFactor;
}
