

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

float random(vec2 co) {
    return fract(dot(co, vec2(3,8)) * dot(co.yx, vec2(7,5)) * 0.03);
}

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
    float f = distToEdgeSq;

    /// all different ways soft shadow strength/delimitations
    //f *= 1500;
    //f = pow(f, 0.8) * 256;
    //f = sqrt(f) * 30;
    
    f = sqrt(f) * 25 - f * 200;

    if(remoteOcclusionDist < f) return 1.0;

    f *= 1 - localOcclusionDist * 128;

    //if(f < 0.8)
    //{
    //    float noise = random(gl_FragCoord.xy);
    //    float noiseStrength = 0.2;
    //    f *= (1.0 + noise * noiseStrength - noiseStrength * 0.5);
    //}

    return f;
}

const float CENTER_WEIGHT = 2;
const int OFFSETS_LEN = 4;
const vec2 OFFSETS[OFFSETS_LEN] = vec2[OFFSETS_LEN](
    vec2( 0.7,  -0.7),  vec2(0.7,  0.7),
    vec2( -0.7,  0.7),  vec2( -0.7, -0.7)
);

void fragment() {
    vec3 albedo = fragColor.rgb * texture(texture0, fragTexCoord).rgb;

    // project into shadow‐map UV
    vec3 proj = fragShadowClipSpace.xyz / fragShadowClipSpace.w;
    proj = proj*0.5 + 0.5;
    if (!between(proj.xy, vec2(0.0), vec2(1.0))) {
        finalColor = vec4(albedo, 1.0);
        return;
    }

    float avgShadow = get_shadow(proj);
    float filterRadius = 1.0/float(1024);
    avgShadow *= CENTER_WEIGHT;
    for (int i = 0; i < OFFSETS_LEN; ++i) {
        vec2 offs = OFFSETS[i] * filterRadius;
        avgShadow += get_shadow(vec3(proj.xy + offs, proj.z));
    }
    avgShadow /= float(OFFSETS_LEN + CENTER_WEIGHT);

    float shadow = 0.5 + clamp(0, 1, avgShadow) * 0.5;

    finalColor = vec4(albedo*shadow, 1);
}
