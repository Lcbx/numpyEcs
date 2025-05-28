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

void main() {
    vec3 albedo = fragColor.rgb * texture(texture0, fragTexCoord).rgb;

    // project into shadow‐map UV
    vec3 proj = fragShadowClipSpace.xyz / fragShadowClipSpace.w;
    proj = proj*0.5 + 0.5;
    vec2 shadowUv = proj.xy;
    if (!between(shadowUv, vec2(0.0), vec2(1.0))) {
        finalColor = vec4(albedo, 1.0);
        return;
    }

    float shadowFactor = 1;

    // determine occlusion
    float fragmentDepth = proj.z;
    float occluderDepth = texture(shadowDepthMap, shadowUv).r;

    //float NDotL = dot(fragNormal, lightDir);
    float occlusionDistance = fragmentDepth - occluderDepth;
    if(occlusionDistance > 0.001) shadowFactor = 0.5;
    else {
        vec3 penumbra = texture(shadowPenumbraMap, shadowUv).rgb;
        vec2 penDir = get_dir(penumbra.gb);
        
        float distToEdgeSq = dot(penDir, penDir);

        if(distToEdgeSq < 0.99){
            vec2 occluderCoord = shadowUv + penDir;
            float f = distToEdgeSq;

            // multiply by shadowmap size / max pixel depth 
            // TODO : passs it as uniform
            f *= 1024;

            occluderDepth = texture(shadowDepthMap, occluderCoord).r;
            float fragmentDepth     = proj.z;
            float occlusionDistance = occluderDepth - fragmentDepth;
            f *= 1 + occlusionDistance;

            // sharpening factor
            f *= 30;
            
            f = clamp(0, 1, f);
            shadowFactor *= f;
        }
    }

    shadowFactor = clamp(0.5, 1, shadowFactor);
    
    finalColor = vec4(albedo *shadowFactor, 1);
}
