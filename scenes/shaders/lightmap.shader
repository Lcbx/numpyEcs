

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 matModel;
uniform mat4 mvp;
uniform vec4 colDiffuse; // tint

uniform mat4 lightVP;

varying vec2 fragTexCoord;
varying vec4 fragColor;
varying vec4 fragPos;
varying vec3 fragNormal;
varying vec4 fragShadowClipSpace;

void vertex(){
	fragTexCoord = vertexTexCoord;
	fragColor = vertexColor * colDiffuse;
	fragNormal = normalize(mat3(matModel) * vertexNormal);
	
	vec4 vertex = vec4(vertexPosition, 1.0);
	fragPos = mvp*vertex;
	gl_Position = fragPos;
	
	fragShadowClipSpace = lightVP*matModel*vertex;
}

uniform sampler2D texture0;			// diffuse

uniform vec3 lightDir;
uniform float shadowSamplingRadius;
uniform sampler2D shadowDepthMap;	   // classic depth map (R channel)
uniform sampler2D shadowPenumbraMap;   // RGB: [meshId, distX, distY]
uniform sampler2D ambientOcclusionMap; // R: intensity

out vec4 finalColor;

float random(vec2 co) {
	return fract(dot(co, vec2(3,8)) * dot(co.yx, vec2(7,5)) * 0.03);
}

float interleavedGradientNoise(vec2 co){
	return fract(52.9829189 * fract(dot(co, vec2(0.06711056, 0.00583715))));
}

bool between(vec2 v, vec2 bottomLeft, vec2 topRight){
	vec2 s = step(bottomLeft, v) - step(topRight, v);
	return bool(s.x * s.y);
}

vec2 get_dir(vec2 encoded){
	return encoded * 2.0 - 1.0;
}

// TODO : maybe put shadows into it's own buffer like occlusion ?
// or put both in the same buffer ? food for thoughts
float get_shadow(vec2 uv, float fragmentDepth){

	//vec3 penumbra = texture(shadowPenumbraMap, uv).rgb;
	//float occluderDepth = penumbra.r;

	float occluderDepth = texture(shadowDepthMap, uv).r;
	float localOcclusionDist = fragmentDepth - occluderDepth;

	if(localOcclusionDist > 0) return 0.0;

	vec3 penumbra = texture(shadowPenumbraMap, uv).rgb;
	vec2 penDir = get_dir(penumbra.gb);

	vec2 remoteCoord = uv + penDir;

	//float remoteOccluderDepth = texture(shadowPenumbraMap, remoteCoord).r;
	float remoteOccluderDepth = texture(shadowDepthMap, remoteCoord).r;
	float remoteOcclusionDist = fragmentDepth - remoteOccluderDepth;

	if(remoteOcclusionDist < 0) return 1.0;

	float distToEdgeSq = dot(penDir, penDir);
	float f = distToEdgeSq;

	f = sqrt(f);

	// causes artifacts
	if(remoteOcclusionDist < f) return 1.0;

	float occlusionFactor = 1.5 - remoteOcclusionDist * 2.0;
	f *= occlusionFactor;
	f *= 120.0; // pass the inverse of this as uniform named shadow blur factor ?

	return f;
}

float randAngle()
{
	ivec2 uv = ivec2(gl_FragCoord.xy);
	float angle = 0;
	angle += 30u * uv.x ^ uv.y + 10u * uv.x * uv.y;
	float ign = interleavedGradientNoise(gl_FragCoord.xy);
	angle += ign * PI;
	angle *= ign;
	return angle;
}

const float POISSON_RADIUS = 3.5;
const int NUM_SAMPLES = 5;
const float INV_NUM_SAMPLES = 1.0 / float(NUM_SAMPLES);
const float NUM_SPIRAL_TURNS = 3;

const float PI =  3.141593;
const float twoPI = 6.283186;

void fragment() {
	vec4 albedo = fragColor;
	//albedo = vec4(1);
	//albedo *= texture(texture0, fragTexCoord);
	
	// 0 = in shadow, 1 = lit
	float shadow = 1;
	
	// project into shadow‐map UV
	vec3 proj = fragShadowClipSpace.xyz / fragShadowClipSpace.w;
	proj = proj*0.5 + 0.5;
	if(between(proj.xy, vec2(0.0), vec2(1.0))){
		float shadowComputed = 0;
		// poisson sampling
		float alpha = 0.5 * INV_NUM_SAMPLES;
		float angle = randAngle();
		float angleInc = NUM_SPIRAL_TURNS * INV_NUM_SAMPLES * twoPI;
		for (int i = 0; i < NUM_SAMPLES; ++i) {
			alpha += INV_NUM_SAMPLES;
			angle += angleInc;
			vec2 disk = vec2(cos(angle), sin(angle)) * alpha;
			shadowComputed += get_shadow(proj.xy + disk * shadowSamplingRadius, proj.z);
		}
		shadow = shadowComputed * INV_NUM_SAMPLES;
	}

	// AO, sampled based on screen uv (from half-res)
	ivec2 viewPx = ivec2(gl_FragCoord.xy * 0.5);
	float occlusion = texelFetch(ambientOcclusionMap, viewPx, 0).r;
	
	// blinn-phong
	vec3 ambient = vec3(0.35 * albedo.rgb);
	ambient *= occlusion;
	vec3 lighting = ambient; 
	
	// TODO : accumulate per light
	// TODO : tonemapping

	// diffuse
	//vec3 lightDir = normalize(light.Position - FragPos);
	float diffuse = max(dot(fragNormal, lightDir), 0.0); // * albedo * light.Color;
	diffuse = min(shadow, diffuse);
	diffuse = mix(0.4, 0.8, clamp(diffuse, 0.0, 1.0));

	// specular
	vec3 viewDir  = normalize(-fragPos.xyz);
	vec3 halfwayDir = normalize(lightDir + viewDir);  
	float specular = pow(max(dot(fragNormal, halfwayDir), 0.0), 8.0);

	// attenuation
	//float dist = length(light.Position - FragPos);
	//float attenuation = 1.0 / (1.0 + light.Linear * dist + light.Quadratic * dist * dist);
	lighting += diffuse * albedo.rgb; // = diffuse * light.Color * attenuation;
	lighting += vec3(specular) * 0.1; // = light.Color * specular * attenuation;
	
	//lighting *= occlusion;

	finalColor = vec4(lighting, albedo.a);
	
	//finalColor = vec4( (0.5 + fragColor.rgb) * occlusion * 0.5, albedo.a);
	//finalColor = vec4( vec3(occlusion) , albedo.a);
	//finalColor = vec4(vec3(1)*shadow, 1);
	//finalColor = vec4(texture(ambientOcclusionMap, viewUV).rgb, 1);
}
