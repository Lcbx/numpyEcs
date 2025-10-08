

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
varying float invSqrtDepthMapSize;

void vertex(){
	fragTexCoord = vertexTexCoord;
	fragColor = vertexColor * colDiffuse;
	fragNormal = normalize(mat3(matModel) * vertexNormal);
	
	vec4 vertex = vec4(vertexPosition, 1.0);
	fragPos = mvp*vertex;
	gl_Position = fragPos;
	
	fragShadowClipSpace = lightVP*matModel*vertex;
	invSqrtDepthMapSize = 1.0 / sqrt(float(textureSize(shadowDepthMap, 0)));
}

uniform sampler2D texture0;			// diffuse

uniform vec3 lightDir;
uniform sampler2D shadowDepthMap;	   // classic depth map (R channel)
//uniform sampler2D shadowPenumbraMap;   // RGB: [meshId, distX, distY]
uniform sampler2D ambientOcclusionMap; // R: intensity

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

float randAngle()
{
	uint x = uint(gl_FragCoord.x);
	uint y = uint(gl_FragCoord.y);
	return (30u * x ^ y + 10u * x * y);
}

const float POISSON_RADIUS = 0.1;
const int NUM_SAMPLES = 8;
const float INV_NUM_SAMPLES = 1.0 / float(NUM_SAMPLES);
const float NUM_SPIRAL_TURNS = float(NUM_SAMPLES/2 + 1);

const float PI =  3.141593;
const float twoPI = 6.283186;


float getDepthMap(vec2 uv, int mip_level) {
	//return textureLod(shadowDepthMap, uv, float(mip_level)).x; // lod level not working ?
	return texelFetch(shadowDepthMap, ivec2(uv * textureSize(shadowDepthMap, mip_level)), mip_level).x;
}

void fragment() {
	vec4 albedo = fragColor;
	//albedo *= texture(texture0, fragTexCoord);
	
	// 0 = in shadow, 1 = lit
	float shadow = 1.0;
	
	// project into shadow‐map UV
	vec3 proj = fragShadowClipSpace.xyz / fragShadowClipSpace.w;
	proj = proj*0.5 + 0.5;

	vec2 uv = proj.xy;
	float fragmentDepth = proj.z;
	
	if (between(uv, vec2(0.0), vec2(1.0))) {

		float occluderDepth = texture(shadowDepthMap, uv).r;
		float occlusionDist = fragmentDepth - occluderDepth;

		// TODO: tap around the pixel (with mipmap ?) to estimate local occlusionDist
		// to determine perspectiveRadius based on it

		if(occlusionDist > 0) {

			// poisson sampling
			float alpha = 0.5 * INV_NUM_SAMPLES;
			float angle = randAngle();
			float angleInc = NUM_SPIRAL_TURNS * INV_NUM_SAMPLES * twoPI;
			float perspectiveRadius = POISSON_RADIUS * invSqrtDepthMapSize;

			for (int i = 0; i < NUM_SAMPLES; ++i) {
				alpha += INV_NUM_SAMPLES;
				angle += angleInc;
				
				vec2 disk = vec2(cos(angle), sin(angle));
				float sqrtOcclusionDist = sqrt(occlusionDist);
				float radius = perspectiveRadius * alpha * (1.0 + sqrtOcclusionDist);
				
				vec2 uv2 = uv + disk * radius;

				if (between(uv2, vec2(0.0), vec2(1.0))){

					float localOccluderDepth = texture(shadowDepthMap, uv2).r;
					float localOcclusionDist = fragmentDepth - localOccluderDepth;

					if(localOcclusionDist > 0) continue;

					float f = localOcclusionDist;
					f = sqrt(f) * 30.0; // pass the inverse of this as uniform named shadow blur factor ?
					//f *= 300.0;
					
					shadow += f / sqrtOcclusionDist;
				}
			}
			shadow *= INV_NUM_SAMPLES;
		}
	}


	// expecting AO to be full size
	ivec2 viewPx = ivec2(gl_FragCoord.xy);
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
