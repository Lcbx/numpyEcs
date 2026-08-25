$input a_position, a_normal, i_data0, i_data1, i_data2, i_data3
$output v_normal, v_color0, v_shadowcoord

#include <bgfx_shader.sh>

uniform mat4 u_lightMtx;

vec3 quatRotate(vec4 q, vec3 v)
{
	vec3 t = cross(q.xyz, v) * 2.0;
	return v + q.w * t + cross(q.xyz, t);
}

void main()
{
	vec3 scale = i_data2.xyz;
	vec3 worldPos = i_data0.xyz + quatRotate(i_data1, a_position * scale);
	vec3 normal = a_normal / scale;
	vec4 world = vec4(worldPos, 1.0);

	gl_Position = mul(u_viewProj, world);
	v_normal = normalize(quatRotate(i_data1, normal));
	v_color0 = i_data3;
	v_shadowcoord = mul(u_lightMtx, world);
}
