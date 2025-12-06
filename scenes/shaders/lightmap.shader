
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
varying float fragDepth;
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
	pos = invProj * pos;
	fragDepth = pow(pos.z * -0.001, 0.3); // poor man's depth conversion (camera's far is 1000)
	fragPos = pos.xyz; // / pos.w;
	
	//vec4 fragShadowClipSpace = lightVP*invVP*fragPos;
	vec4 fragShadowClipSpace = lightVP*matModel*vertex;
	fragShadow = fragShadowClipSpace.xyz / fragShadowClipSpace.w*0.5+0.5;
}


const float PI =  3.141593;
const float twoPI = 6.283186;

// used by shadow map
const int SHADOW_SAMPLES = 3;
const float SHADOW_INV_SAMPLES = SHADOW_SAMPLES > 0 ? 1.0 / float(SHADOW_SAMPLES) : 0;
const float SHADOW_WEIGHT = SHADOW_SAMPLES > 0 ? 0.2 + 0.7 / float(SHADOW_SAMPLES) : 1;
const float SHADOW_SPIRAL_TURNS = (SHADOW_SAMPLES > 3 ? round(SHADOW_SAMPLES * 0.5) + 0.99 : SHADOW_SAMPLES * 0.85 - 0.5);

// used by AO
const int AO_SAMPLES = 2;
const float AO_INV_SAMPLES = AO_SAMPLES > 0 ? 1.0 / float(AO_SAMPLES) : 0;
const float AO_WEIGHT = AO_SAMPLES > 0 ? 0.2 + 0.7 / float(AO_SAMPLES) : 1;
const float AO_SPIRAL_TURNS = (AO_SAMPLES > 3 ? round(AO_SAMPLES * 0.5) + 0.99 : AO_SAMPLES * 0.85 - 0.5);

// used by shadow map
const float POISSON_RADIUS_MULT = 0.1;
const float MIN_POISSON_RADIUS = 0.001;
const float MAX_POISSON_RADIUS = 0.003;

// used by AO
const float AO_radiusWS = 0.04;
const float AO_radiusWS2 = AO_radiusWS * AO_radiusWS;
const float AO_bias = 0.03;
const float AO_intensity = 1.5;

uniform sampler2D texture0; // diffuse

uniform vec3 lightDir;
uniform sampler2D shadowDepthMap; // classic depth map (R channel)
uniform sampler2D ambientOcclusionMap; // R: intensity

uniform mat4 invProj; // inverse proj matrix
uniform sampler2D viewDepthMap;	   // classic depth map (R channel)

float random(vec2 co) {
	return fract(dot(co, ivec2(3,8)) * dot(co.yx, ivec2(7,5)) * 0.03);
}

// NOTE : goes over PI
float randomAngle(ivec2 co){
	return 30u * co.x ^ co.y + 10u * co.x * co.y;
}

float interleavedGradientNoise(ivec2 co){
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
	return interleavedGradientNoise(ivec2(param)) * twoPI;
}

vec3 getPositionVS(vec2 uv, vec2 toPx, int mip_level) {
	float z = texelFetch(viewDepthMap, ivec2(uv * toPx), mip_level).x;
	vec3 clip = vec3(uv,z);
	vec4 view = invProj * vec4(clip, 1);
	return view.xyz / view.w * 0.5 + 0.5;
}

// different AO formulas
float calculateAO(vec3 position, vec3 position2, vec3 normal)
{
	vec3 v = position2 - position;
	
	float vv = dot(v, v);
	float vn = dot(v, normal) - AO_bias;


	//const float epsilon = 0.001;
	
	/// (from the HPG12 paper)
	//return float(vv < AO_radiusWS2) * max(vn / (epsilon + vv), 0.0);
	
	// default / recommended
	//float f = max(AO_radiusWS2 - vv, 0.0) / AO_radiusWS2;
	//return f * f * f * max(vn / (epsilon + vv), 0.0);
	
	return max(1.0 - vv, 0.0) * max(vn, 0.0);
	
	//return float(vv < AO_radiusWS2)* max(vn, 0.0);
}

float sampleAO() {

	vec2 toFullPx = textureSize(viewDepthMap, 0);
	vec2 toFullUv = 1.0 / textureSize(viewDepthMap, 0);
	vec2 uv = gl_FragCoord.xy * toFullUv;

	vec3 position = getPositionVS(uv, toFullPx, 0);
	
	// estimate normal from depth
	vec3 dx = dFdx(position);
	vec3 dy = dFdy(position);
	vec3 normal = normalize(cross(dx, dy));

	//int max_mip = textureQueryLevels(viewDepthMap) - 1;
	int miplevel = 0; //clamp(findMSB(int(radius * 300)), 0, max_mip);

	vec2 toPx = textureSize(viewDepthMap, miplevel);
	vec2 toUv = vec2(1) / toPx;

	float occlusion = 0.0;
	occlusion += calculateAO(position, getPositionVS(uv+ vec2(-5, 0) * toUv, toPx, miplevel), normal);
	occlusion += calculateAO(position, getPositionVS(uv+ vec2(-3, 3) * toUv, toPx, miplevel), normal) * 2.0;
	occlusion += calculateAO(position, getPositionVS(uv+ vec2(0,  5) * toUv, toPx, miplevel), normal);
	occlusion += calculateAO(position, getPositionVS(uv+ vec2(3,  3) * toUv, toPx, miplevel), normal) * 2.0;
	occlusion += calculateAO(position, getPositionVS(uv+ vec2(5,  0) * toUv, toPx, miplevel), normal);
	occlusion += calculateAO(position, getPositionVS(uv+ vec2(3, -3) * toUv, toPx, miplevel), normal) * 2.0;
	occlusion += calculateAO(position, getPositionVS(uv+ vec2(0, -5) * toUv, toPx, miplevel), normal);
	occlusion += calculateAO(position, getPositionVS(uv+ vec2(-3,-3) * toUv, toPx, miplevel), normal) * 2.0;
	occlusion *= 0.0833333333333; // 1/12

	if(AO_INV_SAMPLES > 0)
	{
		float perspectiveRadius = min(AO_radiusWS, AO_radiusWS2 / fragDepth);
		float angle = randAngle(gl_FragCoord.xy);
		float angleInc = AO_SPIRAL_TURNS * AO_INV_SAMPLES * twoPI;
		float radiusInc = AO_INV_SAMPLES * 0.9 * perspectiveRadius;
		float radius = radiusInc;
		
		float total = 0;
		for (int i = 0; i < AO_SAMPLES; ++i) {
			radius += radiusInc;
			angle += angleInc;
			
			vec2 disk = vec2(cos(angle), sin(angle));		
			vec2 uv2 = uv + disk * radius;

			total += calculateAO(position, getPositionVS(uv2, toPx, miplevel), normal);
		}
		occlusion = mix(total * AO_INV_SAMPLES, occlusion, AO_WEIGHT);
	}

	occlusion *= AO_intensity;
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

	float shadow = 0;
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(-2, 0), 0).r);
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(-1, 1), 0).r) * 2.0;
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(0,  2), 0).r);
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(1,  1), 0).r) * 2.0;
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(2,  0), 0).r);
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(1, -1), 0).r) * 2.0;
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(0, -2), 0).r);
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(-1,-1), 0).r) * 2.0;
	shadow *= 0.0833333333333; // 1/12

	if(SHADOW_INV_SAMPLES > 0)
	{
		float distToOccluder = max(0, fragShadow.z - texelFetch(shadowDepthMap, ivec2(uv), 0).r);
		float factor = clamp(POISSON_RADIUS_MULT * distToOccluder, MIN_POISSON_RADIUS, MAX_POISSON_RADIUS);
		
		float angle = randAngle(uv + gl_FragCoord.xy);
		float angleInc = SHADOW_SPIRAL_TURNS * SHADOW_INV_SAMPLES * twoPI;
		vec2 radiusInc = SHADOW_INV_SAMPLES * factor * depthSize;
		vec2 radius = radiusInc;

		float total = 0;
		for (int i = 0; i < SHADOW_SAMPLES; ++i) {
		        radius += radiusInc;
		        angle += angleInc;
		        vec2 disk = vec2(cos(angle), sin(angle));
		        vec2 pixel = uv + disk * radius;
				total += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel), 0).r);
		}
		shadow = mix(total * SHADOW_INV_SAMPLES, shadow, SHADOW_WEIGHT);
	}
	
	return shadow;
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -0.33333333, 0.6666666, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 0.6666666, 0.33333333, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 fastSaturation(vec3 c, float saturation)
{
	return mix(vec3(dot(c, vec3(0.3, 0.7, 0.15)) + 0.2), c, saturation);
}


void fragment() {
	vec4 albedo = fragColor;
	//albedo = vec4(vec3(0.8), 1);
	albedo = vec4(0.9,0.4,0.6,1);
	//albedo *= texture(texture0, fragTexCoord);
	
	// 0 = in shadow, 1 = lit
	float shadow = 1;
	if(between(fragShadow.xy, vec2(0), vec2(1)))
		shadow = tapShadowPoisson();

	// 0 = in shadow, 1 = lit
	float occlusion = 1; 
	if(fragDepth < 0.45)
		occlusion = sampleAO();
	
	// blinn-phong
	vec3 ambient = vec3(0.35 * albedo.rgb);
	ambient *= occlusion;
	//ambient *= 2;

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
	vec3 viewDir  = normalize(-fragPos.xyz);
	vec3 halfwayDir = normalize(lightDir + viewDir);  
	float specular = pow(max(dot(fragNormal, halfwayDir), 0.0), 8.0);
	lighting += vec3(specular) * 0.15; // = light.Color * specular * attenuation;
	
	//lighting *= occlusion;

	// desaturate based on fragment depth 
	float effect = 1.2 - fragDepth;
	//effect = clamp(effect, 0, 1);
	//effect = clamp(effect, 0.5, 1);
	effect = mix(0.4, 1, effect);
	lighting = fastSaturation(lighting, effect);

	// proper hue saturation brightness
	//vec3 hsv = rgb2hsv(lighting);
	//hsv.g *= effect;
	//lighting = hsv2rgb(hsv);

	// debug view
	//lighting = mix(vec3(0.5, 0.5, 0.8), lighting, effect);

	finalColor = vec4(lighting, albedo.a);
	//finalColor = vec4(vec3(fragDepth), albedo.a);
	//finalColor = vec4( (0.5 + fragColor.rgb) * occlusion * 0.5, albedo.a);
	//finalColor = vec4( vec3(occlusion) , albedo.a);
	//finalColor = vec4(vec3(1)*shadow, 1);
	//finalColor = vec4(texture(ambientOcclusionMap, viewUV).rgb, 1);
}
