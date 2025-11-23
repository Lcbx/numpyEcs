

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

void vertex(){
	fragTexCoord = vertexTexCoord;
	fragColor = vertexColor * colDiffuse;
	fragNormal = normalize(mat3(matModel) * vertexNormal);
	
	vec4 vertex = vec4(vertexPosition, 1.0);
	fragPos = mvp*vertex;
	gl_Position = fragPos;
}

uniform sampler2D texture0; // diffuse
uniform sampler2D ambientOcclusionMap; // R: intensity

uniform vec3 lightDir; // directional light

out vec4 finalColor;


float sampleAO() {
	// AO, sampled based on screen uv (from half-res)
	ivec2 viewPx = ivec2(gl_FragCoord.xy * 0.5);
	float occlusion = 0;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(-2, 0), 0).r;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(-1, 1), 0).r * 2.0;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(0, 2), 0).r;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(1, 1), 0).r * 2.0;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(2, 0), 0).r;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(1, -1), 0).r * 2.0;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(0, -2), 0).r;
	occlusion += texelFetch(ambientOcclusionMap, viewPx + ivec2(-1, -1), 0).r * 2.0;
	return occlusion;

}

void fragment() {
	vec4 albedo = fragColor;
	//albedo = vec4(1);
	//albedo *= texture(texture0, fragTexCoord);
	
	// 0 = in shadow, 1 = lit
	float occlusion = sampleAO();	
	
	// blinn-phong
	vec3 ambient = vec3(0.35 * albedo.rgb);
	//ambient *= occlusion;
	vec3 lighting = ambient; 
	
	// TODO : accumulate per light
	// TODO : tonemapping

	// diffuse
	//vec3 lightDir = normalize(light.Position - FragPos);
	float diffuse = max(dot(fragNormal, lightDir), 0.0); // * albedo * light.Color;
	//diffuse = min(shadow, diffuse);
	diffuse = mix(0.4, 0.8, clamp(diffuse, 0.0, 1.0));

	// specular
	vec3 viewDir  = normalize(-fragPos.xyz);
	vec3 halfwayDir = normalize(lightDir + viewDir);  
	float specular = pow(max(dot(fragNormal, halfwayDir), 0.0), 8.0);

	// attenuation
	//float dist = length(light.Position - FragPos);
	//float attenuation = 1.0 / (1.0 + light.Linear * dist + light.Quadratic * dist * dist);
	lighting += diffuse * albedo.rgb; // = diffuse * light.Color * attenuation;
	//lighting += vec3(specular) * 0.1; // = light.Color * specular * attenuation;
	
	lighting *= occlusion;

	finalColor = vec4(lighting, albedo.a);
	
	//finalColor = vec4( (0.5 + fragColor.rgb) * occlusion * 0.5, albedo.a);
	//finalColor = vec4( vec3(occlusion) , albedo.a);
	//finalColor = vec4(vec3(1)*shadow, 1);
	//finalColor = vec4(texture(ambientOcclusionMap, viewUV).rgb, 1);
}
