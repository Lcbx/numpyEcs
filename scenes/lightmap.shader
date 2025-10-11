

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


uniform float invDepthMapSize;
uniform vec3 lightDir;
uniform sampler2D shadowDepthMap;
uniform sampler2D ambientOcclusionMap;

out vec4 finalColor;

bool between(vec2 v, vec2 bottomLeft, vec2 topRight){
	vec2 s = step(bottomLeft, v) - step(topRight, v);
	return bool(s.x * s.y);
}

float randAngle()
{
	uint x = uint(gl_FragCoord.x);
	uint y = uint(gl_FragCoord.y);
	return (30u * x ^ y + 10u * x * y);
}

void fragment() {
	vec4 albedo = fragColor;
	//albedo *= texture(texture0, fragTexCoord);
	

	// 0 = in shadow, 1 = lit
	float shadow = 0;

	// project into shadow‐map UV
	vec3 proj = fragShadowClipSpace.xyz / fragShadowClipSpace.w;
	proj = proj*0.5 + 0.5;
	vec2 uv = proj.xy;
	float fragmentDepth = proj.z;

	if(between(uv, vec2(0.0), vec2(1.0))){

		float localOcclusionDepth = texture(shadowDepthMap, uv).r;

		float angle = randAngle();
		vec2 angleOffset = vec2(cos(angle), sin(angle));

		float weights[8] = float[8](1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0);
		vec2 taps[8] = vec2[8](
			vec2(-2.0, 0.0), vec2(-1.0, 1.0), vec2(0.0, 2.0), vec2(1.0, 1.0),
			vec2(2.0, 0.0), vec2(1.0, -1.0), vec2(0.0, -2.0), vec2(-1.0, -1.0)
		);

		float factor = 5.0 * invDepthMapSize * sqrt(fragmentDepth - localOcclusionDepth);

		for(int i=0;i<12;i++){
			vec2 uv2 = uv + (taps[i] + angleOffset) * factor;
			float occluderDepth = texture(shadowDepthMap, uv2).r;
			shadow += (fragmentDepth <= occluderDepth) ? weights[i] : 0.0;
		}
		shadow *= 0.08333; // 1/12
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
