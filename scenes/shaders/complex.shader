
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
<ProbeUniforms> geometry_bias: f32;
<ProbeUniforms> sample_bias: f32;
<ProbeUniforms> light_sampling: vec4f; // cos(angular radius), visibility alpha, padding


@group(2) @binding(0)
var<uniform> probe_uniforms: ProbeUniforms;

@group(2) @binding(1) var probe_sampler: sampler;
@group(2) @binding(2) var probe_sh0: texture_3d<f32>;
@group(2) @binding(3) var probe_sh1: texture_3d<f32>;
@group(2) @binding(4) var probe_sh2: texture_3d<f32>;
@group(2) @binding(5) var probe_sh3: texture_3d<f32>;

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
	if probe_uniforms.trace.w >= 3u { return vec4f(irradiance, 1.0); }
	return vec4f(color.rgb * irradiance * (1.0 / 3.14159265), color.a);
}


// Cubic B-spline pairs reduce 64 taps to eight hardware trilinear samples.
fn sample_probe_irradiance(position: vec3f, normal: vec3f) -> vec3f {
	let dims = vec3f(probe_uniforms.dimensions.xyz);
	let sample_position = position + normal * probe_uniforms.sample_bias;
	let grid = clamp((sample_position - probe_uniforms.origin.xyz) / probe_uniforms.origin.w, vec3f(0.0), dims - 1.0);
	let base = floor(grid);
	let f = fract(grid);
	let f2 = f * f;
	let f3 = f2 * f;
	let w0 = (1.0 - 3.0*f + 3.0*f2 - f3) / 6.0;
	let w1 = (4.0 - 6.0*f2 + 3.0*f3) / 6.0;
	let w2 = (1.0 + 3.0*f + 3.0*f2 - 3.0*f3) / 6.0;
	let w3 = f3 / 6.0;
	let g0 = w0 + w1;
	let g1 = w2 + w3;
	let uv0 = (base - 0.5 + w1 / g0) / dims;
	let uv1 = (base + 1.5 + w3 / g1) / dims;
	var c0 = vec4f(0.0);
	var c1 = vec4f(0.0);
	var c2 = vec4f(0.0);
	var c3 = vec4f(0.0);
	for (var corner = 0u; corner < 8u; corner++) {
		let upper = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u) != vec3u(0u);
		let uv = select(uv0, uv1, upper);
		let weights = select(g0, g1, upper);
		let weight = weights.x * weights.y * weights.z;
		c0 += textureSampleLevel(probe_sh0, probe_sampler, uv, 0.0) * weight;
		c1 += textureSampleLevel(probe_sh1, probe_sampler, uv, 0.0) * weight;
		c2 += textureSampleLevel(probe_sh2, probe_sampler, uv, 0.0) * weight;
		c3 += textureSampleLevel(probe_sh3, probe_sampler, uv, 0.0) * weight;
	}
	if c0.a <= 0.0 { return vec3f(0.0); }
	if probe_uniforms.trace.w == 3u { return vec3f(1.0); }
	if probe_uniforms.trace.w == 4u { return vec3f(c2.a / c0.a); }
	let basis = sh_basis(normal);
	let bounce = max((c0.rgb * basis.x + c1.rgb * basis.y + c2.rgb * basis.z + c3.rgb * basis.w) / c0.a, vec3f(0.0));
	let visibility = clamp(c1.a / c0.a, 0.0, 1.0);
	let light_dir = normalize(probe_uniforms.light_direction.xyz);
	let direct = visibility * probe_uniforms.light_radiance.rgb * max(dot(normal, light_dir), 0.0);
	if probe_uniforms.trace.w == 0u { return direct; }
	if probe_uniforms.trace.w == 1u { return bounce; }
	return direct + bounce;
}
