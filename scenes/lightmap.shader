

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;

uniform mat4 lightVP;

varying vec2 fragTexCoord;
varying vec4 fragColor;
varying vec3 fragNormal;
varying vec4 fragShadowClipSpace;

void vertex(){
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = normalize(mat3(matModel) * vertexNormal);
    
    vec4 vertex = vec4(vertexPosition, 1.0);
    gl_Position = mvp*vertex;
    
    fragShadowClipSpace = lightVP*matModel*vertex;
}

uniform sampler2D texture0;            // diffuse

uniform vec3 lightDir;
uniform sampler2D shadowDepthMap;      // classic depth map (R channel)
uniform sampler2D shadowPenumbraMap;   // RGB: [meshID, distX, distY]

out vec4 finalColor;

bool between(vec2 v, vec2 bottomLeft, vec2 topRight){
    vec2 s = step(bottomLeft, v) - step(topRight, v);             
	return bool(s.x * s.y);
}

vec2 get_dir(vec2 encoded){
    return encoded * 2.0 - 1.0;
}

float get_shadow(vec3 proj){

    // determine occlusion
    float fragmentDepth = proj.z;

    float occluderDepth = texture(shadowDepthMap, proj.xy).r;
    float localOcclusionDist = fragmentDepth - occluderDepth;

    if(localOcclusionDist > 0) return 0.0;

    vec3 penumbra = texture(shadowPenumbraMap, proj.xy).rgb;
    vec2 penDir = get_dir(penumbra.gb);

    vec2 remoteCoord = proj.xy + penDir;

    float remoteOccluderDepth = texture(shadowDepthMap, remoteCoord).r;
    float remoteOcclusionDist = fragmentDepth - remoteOccluderDepth;

    if(remoteOcclusionDist < 0) return 1.0;

    float distToEdgeSq = dot(penDir, penDir);
    float f = distToEdgeSq * 1024;

    if(remoteOcclusionDist < f) return 1.0;

    f *= 1 - localOcclusionDist * 1024;

    //float noise = random(gl_FragCoord.xy);
    //float noiseStrength = 0.25;
    //f *= (1.0 + noise * noiseStrength - noiseStrength);

    return f;
}

void fragment() {
    vec3 albedo = fragColor.rgb * texture(texture0, fragTexCoord).rgb;

    // project into shadow‐map UV
    vec3 proj = fragShadowClipSpace.xyz / fragShadowClipSpace.w;
    proj = proj*0.5 + 0.5;
    if (!between(proj.xy, vec2(0.0), vec2(1.0))) {
        finalColor = vec4(albedo, 1.0);
        return;
    }

    float shadow = 0.5 + clamp(0, 1, get_shadow(proj) ) * 0.5;

    finalColor = vec4(albedo*shadow, 1);
}
