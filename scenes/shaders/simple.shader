
in vec3 vPos;
in vec3 vNormal;
in vec2 vUV;

// from instance buffer
in vec3 iPosition;
in vec4 iRotation; // quaternion
in vec4 iScale;    // last float16 is unused
in uint iTint;

uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uInverseProj;

uniform vec3 uLightDir;
uniform mat4 uLightViewProj;
uniform texture2D uLightMap; // classic depth map (R channel)

varying vec3 vViewPos;
varying vec3 vNormalWS;
varying vec2 vUV;
flat varying uint tint;
varying vec3 vShadowPos;


vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) * 2.0;
    return v + q.w * t + cross(q.xyz, t);
}

bool between(vec2 v, vec2 bottomLeft, vec2 topRight){
	vec2 s = step(bottomLeft, v) - step(topRight, v);
	return bool(s.x * s.y);
}

void vertex() {
    vec3 worldPos = iPosition + quat_rotate(iRotation, vPos * iScale.xyz);

    vec3 normal = vNormal;
    if(iScale.xyz!=vec3(1.0,1.0,1.0)) normal /= iScale.xyz;
    vec3 world_normal = quat_rotate(iRotation, normal);
    vNormalWS = normalize(world_normal);

	vec4 vertex = vec4(worldPos, 1.0);
    vec4 view_pos = uView * vertex;
    gl_Position = uProj * view_pos;

    vViewPos.xyz = view_pos.xyz;
	vec4 shadow_clip = uLightViewProj*vertex;
    vShadowPos = shadow_clip.xyz / shadow_clip.w *0.5+0.5;

    vUV = vUV;
    tint = iTint;
}

// TODO: move color utilities into a shader_include file

float srgb_to_linear_channel(float c) {
    if( c <= 0.04045 ){
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

vec4 unpack_rgba8_srgb(uint c) {
    vec4 rgba = vec4(
        float( c         & 255u),
        float((c >> 8u)  & 255u),
        float((c >> 16u) & 255u),
        float((c >> 24u) & 255u)
    ) / 255.0;

    return vec4(
        srgb_to_linear_channel(rgba.r),
        srgb_to_linear_channel(rgba.g),
        srgb_to_linear_channel(rgba.b),
        rgba.a
    );
}

vec3 fastSaturation(vec3 c, float saturation)
{
	return mix(vec3(dot(c, vec3(0.3, 0.7, 0.15)) + 0.2), c, saturation);
}

out vec4 FragColor;

void fragment() {
    vec4 color = unpack_rgba8_srgb(tint);
    
	float shadow = 1;
	//if(between(fragShadow.xy, vec2(0), vec2(1)))
	//	shadow = tapShadowPoisson();

    float NdotL = max(dot(vNormalWS, uLightDir), 0.0);
    vec3 lighting = color.rgb * mix(0.4, 0.8, min(shadow, NdotL));

	vec3 halfwayDir = normalize(uLightDir - vViewPos.xyz);  
	float specular = pow(max(dot(vNormalWS, halfwayDir), 0.0), 8.0);
	//lighting += vec3(specular) * 0.15; // = light.Color * specular * attenuation;

	// desaturate based on fragment depth 
	//float effect = 1.2 - vViewPos.z;
	//effect = mix(0.4, 1.0, effect);
	//lighting = fastSaturation(lighting, effect);

    FragColor = vec4(lighting, color.a);
}
