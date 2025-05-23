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
    float mapDepth = texture(shadowDepthMap, shadowUv).r;

    // bias to avoid self‐shadow acne
    float NDotL = max(dot(fragNormal, lightDir), 0.0);
    float bias  = 0.001 * (1 - NDotL);

    // determine occlusion
    bool occluded = proj.z > mapDepth + bias;

    float shadowFactor = 1;
    if (occluded) {
        shadowFactor = 0.5;
    }
    else {
        vec3  penumbra = texture(shadowPenumbraMap, shadowUv).rgb;
        vec2 penDir = penumbra.gb;
        penDir = penDir * 2.0 - 1.0;
        float distToEdgeSq = dot(penDir, penDir);
        float f = distToEdgeSq;

        // TODO: we might need another sample to make a better shadow gradient
        // where to sample though ? midway to dir ? away ?

        // sqrt(2) = 1.42
        f = smoothstep(0, 1.4, f);

        //float noise = random(gl_FragCoord.xy);
        //float noiseStrength = 0.3;
        //if(f < 0.99) f *= (1.0 + noise * noiseStrength - noiseStrength);
        
        shadowFactor = mix(0.5, 1, f);
    }
    finalColor = vec4(albedo * shadowFactor, 1.0);
    //finalColor = vec4(penumbra * shadowFactor, 1);
}
