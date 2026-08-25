$input a_position, i_data0, i_data1, i_data2

#include <bgfx_shader.sh>

vec3 quatRotate(vec4 q, vec3 v)
{
	vec3 t = cross(q.xyz, v) * 2.0;
	return v + q.w * t + cross(q.xyz, t);
}

void main()
{
	vec3 worldPos = i_data0.xyz + quatRotate(i_data1, a_position * i_data2.xyz);
	gl_Position = mul(u_viewProj, vec4(worldPos, 1.0));
}
