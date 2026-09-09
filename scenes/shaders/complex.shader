
<Uniforms> view: mat4x4f;
<Uniforms> proj: mat4x4f;
<Uniforms> light_dir: vec4f;
<Uniforms> light_view_proj: mat4x4f;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

<MeshInstance> iPosition: vec3f;
<MeshInstance> iTint: u32;
<MeshInstance> iRotation: vec2u;
<MeshInstance> iScale: vec2u;

@group(1) @binding(0)
var<storage, read> instances: array<MeshInstance>;

@group(1) @binding(1)
var<storage, read> visible_instances: array<u32>;

<VertexInput> position: vec3f;
<VertexInput> normal: vec3f;
<VertexInput> uv: vec2f;

<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> normal: vec3f;
<VertexOutput> world_position: vec3f;
<VertexOutput> @interpolate(flat) tint: u32;


<ProbeUniforms> origin: vec4f;
<ProbeUniforms> dimensions: vec4u;
<ProbeUniforms> light_direction: vec4f;
<ProbeUniforms> light_radiance: vec4f;
<ProbeUniforms> trace: vec4u;
<ProbeUniforms> update_count: u32;
<ProbeUniforms> frame_index: u32;
<ProbeUniforms> padding: vec2u;

<Probe> direct:   array<vec4f, 4>;
<Probe> bounce:   array<vec4f, 4>;
<Probe> metadata: vec4u;

@group(2) @binding(0)
var<uniform> probe_uniforms: ProbeUniforms;

@group(2) @binding(1)
var<storage, read> probes: array<Probe>;

const SH0 = 0.28209479;
const SH1 = 0.48860251;

fn sh_basis(direction: vec3f) -> vec4f {
	return vec4f(SH0, SH1 * direction.y, SH1 * direction.z, SH1 * direction.x);
}


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
	output.world_position = trans.world_pos;
	output.tint = inst.iTint;
	return output;
}
@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
	let color = unpack_rgba8_srgb(input.tint);
	let irradiance = sample_probe_irradiance(input.world_position, normalize(input.normal));
	return vec4f(color.rgb * irradiance * (1.0 / 3.14159265), color.a);
}


fn probe_linear_index(coord: vec3u) -> u32 {
	let dims = probe_uniforms.dimensions.xyz;
	return coord.x + dims.x * (coord.y + dims.y * coord.z);
}

fn sample_probe_irradiance(position: vec3f, normal: vec3f) -> vec3f {
	let grid = clamp((position - probe_uniforms.origin.xyz) / probe_uniforms.origin.w, vec3f(0.0), vec3f(probe_uniforms.dimensions.xyz) - 1.0001);
	let base = vec3u(floor(grid));
	let fraction = fract(grid);
	let basis = sh_basis(normal);
	var result = vec3f(0.0);
	var weight_sum = 0.0;
	for (var corner = 0u; corner < 8u; corner++) {
		let offset = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
		let coord = min(base + offset, probe_uniforms.dimensions.xyz - 1u);
		let probe = probes[probe_linear_index(coord)];
		if probe.metadata.x == 0u { continue; }
		let selector = vec3f(offset);
		let weights = select(vec3f(1.0) - fraction, fraction, selector == vec3f(1.0));
		let weight = weights.x * weights.y * weights.z;
		var irradiance = vec3f(0.0);
		for (var band = 0u; band < 4u; band++) {
			if probe_uniforms.trace.w != 1u { irradiance += probe.direct[band].rgb * basis[band]; }
			if probe_uniforms.trace.w != 0u { irradiance += probe.bounce[band].rgb * basis[band]; }
		}
		if probe_uniforms.trace.w == 3u { irradiance = vec3f(1.0); }
		if probe_uniforms.trace.w == 4u { irradiance = vec3f(min(log2(f32(probe.metadata.y) + 1.0) / 8.0, 1.0)); }
		result += max(irradiance, vec3f(0.0)) * weight;
		weight_sum += weight;
	}
	return select(vec3f(0.0), result / weight_sum, weight_sum > 0.0);
}
