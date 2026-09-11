#include "draw_common.shaderlib"
#from "draw_common.shaderlib" import instance_bindings
{{ instance_bindings() }}

<VertexInput> position: vec3f;
<VertexInput> normal: vec3f;
<VertexInput> uv: vec2f;

<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> normal: vec3f;
<VertexOutput> @interpolate(flat) tint: u32;


#include "utils.shaderlib"
#from "utils.shaderlib" import instance_transform_unpacking, unpack_srgb_color
{{ instance_transform_unpacking(InstanceType="MeshInstance") }}
{{ unpack_srgb_color() }}

@vertex
fn vertex(input: VertexInput, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
	let real_id = visible_instances[instance_idx];
	let inst = instances[real_id];

	let trans = unpack_instance_transform(input.position, inst);

	var normal = input.normal;
	if any(trans.scale != vec3f(1.0)) {
		normal /= trans.scale;
	}

	let normal_ws = normalize(quat_rotate(trans.rotation, normal));
	let world = vec4f(trans.world_pos, 1.0);
	let view_pos = uniforms.view * world;

	var output: VertexOutput;
	output.position = uniforms.proj * view_pos;
	output.normal = normal_ws;
	output.tint = inst.iTint;
	return output;
}
@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
	let color = unpack_rgba8_srgb(input.tint);
	let light = max(dot(input.normal, uniforms.light_dir.xyz), 0.0);
	let lighting = mix(0.35, 1.0, light);
	return vec4f(color.rgb * lighting, color.a);
}
