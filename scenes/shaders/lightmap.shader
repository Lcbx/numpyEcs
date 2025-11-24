
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;
uniform mat4 matModel;
uniform vec4 colDiffuse; // tint

uniform mat4 lightVP;

uniform sampler2D texture0; // diffuse

uniform vec3 lightDir;
uniform sampler2D shadowDepthMap;	   // classic depth map (R channel)
uniform sampler2D ambientOcclusionMap; // R: intensity

varying vec2 fragTexCoord;
varying vec4 fragColor;
varying vec4 fragPos;
varying vec3 fragNormal;
varying float fragDepth;
varying vec3 fragShadow;


const float PI =  3.141593;
const float twoPI = 6.283186;

void vertex(){
	fragTexCoord = vertexTexCoord;
	fragColor = vertexColor * colDiffuse;
	fragNormal = normalize(mat3(matModel) * vertexNormal);
	
	vec4 vertex = vec4(vertexPosition, 1.0);
	//fragPos = matProjection*matView*matModel*vertex;
	fragPos = mvp*vertex;
	gl_Position = fragPos;
	fragDepth = (fragPos.z / fragPos.w) *0.5 + 0.5;
	
	//vec4 fragShadowClipSpace = lightVP*invVP*fragPos;
	vec4 fragShadowClipSpace = lightVP*matModel*vertex;
	fragShadow = (fragShadowClipSpace.xyz / fragShadowClipSpace.w) *0.5 + 0.5;
}


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

float randAngle(vec2 param)
{
	ivec2 uv = ivec2(param);
	float angle = 0;
	angle = 30u * uv.x ^ uv.y + 10u * uv.x * uv.y;
	//angle = interleavedGradientNoise(uv);
	return angle;
}


float sampleAO() {
	// AO, sampled based on screen uv (from half-res)
	ivec2 viewPx = ivec2(gl_FragCoord.xy * 0.5);
	float occlusion = 0;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(-2,  0), 0).r;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(-1,  1), 0).r * 2.0;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(0,   2), 0).r;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(1,   1), 0).r * 2.0;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(2,   0), 0).r;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(1,  -1), 0).r * 2.0;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(0,  -2), 0).r;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(-1, -1), 0).r * 2.0;
	occlusion *= 0.0833333333333; // 1/12
	return occlusion;
}



const float POISSON_RADIUS = 2.5;
const int NUM_SAMPLES = 4;
const float INV_NUM_SAMPLES = 1.0 / float(NUM_SAMPLES);
const float NUM_SPIRAL_TURNS = 3;


// NOTE : try moving ao with shadow sampling ?
float tapShadowPoisson(vec2 step, float startAngle) {
	float shadow = 0;
	vec2 uv = fragShadow.xy + step;
	vec2 depthSize = textureSize(shadowDepthMap,0);
	// poisson sampling
	float alpha = 0.5 * INV_NUM_SAMPLES;
	float angle = startAngle; // randAngle(gl_FragCoord.xy* (1-step) );
	float angleInc = NUM_SPIRAL_TURNS * INV_NUM_SAMPLES * twoPI;
	for (int i = 0; i < NUM_SAMPLES; ++i) {
	        alpha += INV_NUM_SAMPLES;
	        angle += angleInc;
	        vec2 disk = vec2(cos(angle), sin(angle)) * alpha * POISSON_RADIUS;
	        vec2 pixel = (uv * depthSize + disk);
	        shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(pixel), 0).r);
	}
	shadow *= INV_NUM_SAMPLES;
	return shadow;
}


float tapShadowSimple(vec2 step) {
	float shadow = 0;
	vec2 depthSize = textureSize(shadowDepthMap,0);
	vec2 uv = (fragShadow.xy + step) * depthSize;
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(-2, 0), 0).r);
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(-1, 1), 0).r) * 2.0;
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(0,  2), 0).r);
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(1,  1), 0).r) * 2.0;
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(2,  0), 0).r);
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(1, -1), 0).r) * 2.0;
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(0, -2), 0).r);
	shadow += float(fragShadow.z < texelFetch(shadowDepthMap, ivec2(uv) + ivec2(-1,-1), 0).r) * 2.0;
	shadow *= 0.0833333333333; // 1/12
	return shadow;
}



float sampleShadows()
{
	float shadow = 0.0;
	//shadow += tapShadowPoisson( vec2(0) );
	float factor = 2100 * (1 - fragDepth);
	vec2 stepX = dFdx(fragShadow.xy) * factor;
	vec2 stepY = dFdy(fragShadow.xy) * factor;
	float startAngle = randAngle(gl_FragCoord.xy);
	shadow += tapShadowPoisson( - stepX.xy, startAngle );
	shadow += tapShadowPoisson(   stepX.xy, startAngle );
	shadow += tapShadowPoisson( - stepY.xy, startAngle );
	shadow += tapShadowPoisson(   stepY.xy, startAngle );
	//shadow *= INV_NUM_SAMPLES;
	//shadow *= 0.5; // 1/2
	shadow *= 0.25; // 1/4
	//shadow *= 0.2; // 1/5
	return shadow;
}


void fragment() {
	vec4 albedo = fragColor;
	//albedo = vec4(1);
	//albedo *= texture(texture0, fragTexCoord);
	
	// 0 = in shadow, 1 = lit
	float shadow = 1;
	if(between(fragShadow.xy, vec2(0), vec2(1)))
		shadow = sampleShadows();

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
