
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;
uniform mat4 matModel;
uniform vec4 colDiffuse; // tint

uniform mat4 lightVP;

varying vec2 fragTexCoord;
varying vec4 fragColor;
varying vec3 fragPos;
varying vec3 fragNormal;
varying vec3 fragShadow;

out vec4 finalColor;


void vertex(){
	fragTexCoord = vertexTexCoord;
	fragColor = vertexColor * colDiffuse;
	fragNormal = normalize(mat3(matModel) * vertexNormal);
	
	vec4 vertex = vec4(vertexPosition, 1.0);
	vec4 pos = mvp*vertex;
	gl_Position = pos;
	fragPos = (pos.xyz / pos.w) *0.5 + 0.5;
	
	//vec4 fragShadowClipSpace = lightVP*invVP*fragPos;
	vec4 fragShadowClipSpace = lightVP*matModel*vertex;
	fragShadow = (fragShadowClipSpace.xyz / fragShadowClipSpace.w) *0.5 + 0.5;
}


const float PI =  3.141593;
const float twoPI = 6.283186;

// used by shadow map and AO
const int NUM_SAMPLES = 3;
const float INV_NUM_SAMPLES = 1.0 / float(NUM_SAMPLES);
const float NUM_SPIRAL_TURNS = (NUM_SAMPLES > 3 ? round(NUM_SAMPLES * 0.5) + 0.99 : NUM_SAMPLES * 0.85 - 0.5);

// used by shadow map
const float POISSON_RADIUS_MULT = 10;
const float MIN_POISSON_RADIUS = 0.001;
const float MAX_POISSON_RADIUS = 0.003;

// used by AO
const float radiusWS = 0.06;
const float radiusWS2 = radiusWS * radiusWS;
const float invRadius2 = 1.0 / radiusWS2;
const float bias = 0.05;
const float intensity = 3.0;

uniform sampler2D texture0; // diffuse

uniform vec3 lightDir;
uniform sampler2D shadowDepthMap; // classic depth map (R channel)
uniform sampler2D ambientOcclusionMap; // R: intensity

uniform mat4 invProj; // inverse proj matrix
uniform sampler2D viewDepthMap;	   // classic depth map (R channel)

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

float randAngle(vec2 param)
{
	ivec2 uv = ivec2(param);
	float angle = 0;
	//angle += 30u * uv.x ^ uv.y + 10u * uv.x * uv.y;
	angle += interleavedGradientNoise(uv) * twoPI;
	return angle;
}

vec3 getPositionVS(ivec2 pixel, vec2 toUv, int mip_level) {
	float z = texelFetch(viewDepthMap, pixel, mip_level).x;
	vec3 clip = vec3(vec2(pixel)*toUv,z)*2.0 - vec3(1.0);
	vec4 view = invProj * vec4(clip, 1);
	return view.xyz / view.w * 0.5 + 0.5;
}

// different AO formulas
float calculateAO(vec3 position, vec3 position2, vec3 normal)
{
	vec3 v = position2 - position;
	
	float vv = dot(v, v);
	float vn = dot(v, normal) - bias;


	const float epsilon = 0.001;
	
	/// (from the HPG12 paper)
	//return float(vv < radiusWS2) * max(vn / (epsilon + vv), 0.0);
	
	// default / recommended
	//float f = max(radiusWS2 - vv, 0.0) / radiusWS2;
	//return f * f * f * max(vn / (epsilon + vv), 0.0);
	
	return max(1.0 - vv, 0.0) * max(vn, 0.0);
	
	//return float(vv < radiusWS2)* max(vn, 0.0);
}

float sampleAO() {

	vec2 toFullUv = 1.0 / textureSize(viewDepthMap, 0);
	vec2 uv = gl_FragCoord.xy * toFullUv;

	vec3 position = getPositionVS(ivec2(gl_FragCoord.xy), toFullUv, 0);
	
	// estimate normal from depth
	vec3 dx = dFdx(position);
	vec3 dy = dFdy(position);
	vec3 normal = normalize(cross(dx, dy));

	float perspectiveRadius = radiusWS / position.z;
	float angle = randAngle(gl_FragCoord.xy);
	float angleInc = NUM_SPIRAL_TURNS * INV_NUM_SAMPLES * twoPI;
	float radiusInc = INV_NUM_SAMPLES * perspectiveRadius;
	float radius = 0.5 * radiusInc;
	
	//int max_mip = textureQueryLevels(viewDepthMap) - 1;
	int miplevel = 0; //clamp(findMSB(int(radius * 300)), 0, max_mip);

	vec2 fromUv = textureSize(viewDepthMap, miplevel);
	vec2 toUv = vec2(1) / fromUv;
	
	float occlusion = 0.0;
	for (int i = 0; i < NUM_SAMPLES; ++i) {
		radius += radiusInc;
		angle += angleInc;
		
		vec2 disk = vec2(cos(angle), sin(angle));		
		vec2 uv2 = uv + disk * radius;

		float total = 0;
		//ivec2 viewPx2 = ivec2(uv2 * fromUv);
		ivec2 viewPx2 = ivec2(uv2 * fromUv);
		total += calculateAO(position, getPositionVS( viewPx2, toUv, miplevel), normal);
		occlusion += total;
	}
	occlusion *= INV_NUM_SAMPLES;
	occlusion *= intensity;
	occlusion = 1.0 - occlusion;
	occlusion = clamp(occlusion, 0.0, 1.0);

	return occlusion;
}


// TODO: try moving ao with shadow sampling ?
// TODO: try using mipLevel ?
// TODO: decide whether its useful to use disk samplinmg based on frag depth
float tapShadowPoisson() {
	vec2 depthSize = textureSize(shadowDepthMap,0);
	vec2 uv = fragShadow.xy * depthSize;

	//if(between(uv, vec2(0), vec2(1))) return 1;

	float distToOccluder = max(0, fragShadow.z - texelFetch(shadowDepthMap, ivec2(uv), 0).r);
	float invFragDepth = 1.0 - fragPos.z;
	float factor = clamp(POISSON_RADIUS_MULT * invFragDepth * distToOccluder, MIN_POISSON_RADIUS, MAX_POISSON_RADIUS);
	
	float shadow = 0;
	float angle = randAngle(uv + gl_FragCoord.xy);
	float angleInc = NUM_SPIRAL_TURNS * INV_NUM_SAMPLES * twoPI;
	vec2 radiusInc = INV_NUM_SAMPLES * factor * depthSize;
	vec2 radius = 0.5 * radiusInc;

	for (int i = 0; i < NUM_SAMPLES; ++i) {
	        radius += radiusInc;
	        angle += angleInc;
	        vec2 disk = vec2(cos(angle), sin(angle));
	        vec2 pixel = uv + disk * radius;
			float total = 0;
			//total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel), 0).r);
			total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel) + ivec2(-2, 0), 0).r);
			total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel) + ivec2(-1, 1), 0).r) * 2.0;
			total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel) + ivec2(0,  2), 0).r);
			total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel) + ivec2(1,  1), 0).r) * 2.0;
			total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel) + ivec2(2,  0), 0).r);
			total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel) + ivec2(1, -1), 0).r) * 2.0;
			total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel) + ivec2(0, -2), 0).r);
			total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel) + ivec2(-1,-1), 0).r) * 2.0;
			total*= 0.0833333333333; // 1/12
			shadow += total;
	}
	shadow *= INV_NUM_SAMPLES;
	return shadow;
}


void fragment() {
	vec4 albedo = fragColor;
	//albedo = vec4(1);
	//albedo *= texture(texture0, fragTexCoord);
	
	// 0 = in shadow, 1 = lit
	float shadow = 1;
	if(between(fragShadow.xy, vec2(0), vec2(1)))
		shadow = tapShadowPoisson();

	// 0 = in shadow, 1 = lit
	float occlusion = sampleAO();	
	
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

	// attenuation
	//float dist = length(light.Position - FragPos);
	//float attenuation = 1.0 / (1.0 + light.Linear * dist + light.Quadratic * dist * dist);
	lighting += diffuse * albedo.rgb; // = diffuse * light.Color * attenuation;

	// specular
	//vec3 viewDir  = normalize(-fragPos.xyz);
	//vec3 halfwayDir = normalize(lightDir + viewDir);  
	//float specular = pow(max(dot(fragNormal, halfwayDir), 0.0), 8.0);
	//lighting += vec3(specular) * 0.1; // = light.Color * specular * attenuation;
	
	//lighting *= occlusion;

	finalColor = vec4(lighting, albedo.a);
	
	//finalColor = vec4( (0.5 + fragColor.rgb) * occlusion * 0.5, albedo.a);
	//finalColor = vec4( vec3(occlusion) , albedo.a);
	//finalColor = vec4(vec3(1)*shadow, 1);
	//finalColor = vec4(texture(ambientOcclusionMap, viewUV).rgb, 1);
}
