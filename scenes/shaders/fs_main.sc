$input v_normal, v_color0, v_shadowcoord

#include <bgfx_shader.sh>

SAMPLER2DSHADOW(s_shadowMap, 0);
uniform vec4 u_lightDir;

float srgbToLinearChannel(float c)
{
	if (c <= 0.04045)
	{
		return c / 12.92;
	}
	return pow((c + 0.055) / 1.055, 2.4);
}

vec4 srgbToLinear(vec4 color)
{
	return vec4(
		srgbToLinearChannel(color.r),
		srgbToLinearChannel(color.g),
		srgbToLinearChannel(color.b),
		color.a
	);
}

float sampleShadow(vec4 shadowCoord)
{
	vec3 position = shadowCoord.xyz / shadowCoord.w;
	bool outside = any(lessThan(position.xy, vec2_splat(0.0)))
		|| any(greaterThan(position.xy, vec2_splat(1.0)))
		|| position.z < 0.0
		|| position.z > 1.0;

	if (outside)
	{
		return 1.0;
	}

	return shadow2D(s_shadowMap, vec3(position.xy, position.z - 0.001));
}

void main()
{
	vec4 color = srgbToLinear(v_color0);
	float light = max(dot(v_normal, u_lightDir.xyz), 0.0);
	float shadow = sampleShadow(v_shadowcoord);
	float lighting = mix(0.35, 1.0, light) * mix(0.45, 1.0, shadow);
	gl_FragColor = vec4(color.rgb * lighting, color.a);
}
