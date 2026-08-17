
struct Uniforms {
	view: mat4x4<f32>,
	proj: mat4x4<f32>,
	light_dir: vec4<f32>,
	light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var shadow_map: texture_depth_2d;

@group(1) @binding(1)
var shadow_sampler: sampler_comparison;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) normal: vec3<f32>,
	@location(2) uv: vec2<f32>,
	@location(3) iPosition: vec3<f32>,
	@location(4) iRotation: vec4<f32>,
	@location(5) iScale: vec4<f32>,
	@location(6) iTint: u32,
};

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) normal: vec3<f32>,
	@location(1) @interpolate(flat) tint: u32,
	@location(2) shadow_pos: vec3<f32>,
};

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
	let t = cross(q.xyz, v) * 2.0;
	return v + q.w * t + cross(q.xyz, t);
}

fn world_position(
	position: vec3<f32>,
	iPosition: vec3<f32>,
	iRotation: vec4<f32>,
	iScale: vec3<f32>,
) -> vec3<f32> {
	return iPosition + quat_rotate(iRotation, position * iScale);
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
	let world_pos = world_position(
		input.position,
		input.iPosition,
		input.iRotation,
		input.iScale.xyz,
	);

	var normal = input.normal;
	if any(input.iScale.xyz != vec3<f32>(1.0)) {
		normal /= input.iScale.xyz;
	}

	let normal_ws = normalize(quat_rotate(input.iRotation, normal));
	let world = vec4<f32>(world_pos, 1.0);
	let view_pos = uniforms.view * world;
	let shadow_clip = uniforms.light_view_proj * world;
	let shadow_ndc = shadow_clip.xyz / shadow_clip.w;

	var output: VertexOutput;
	output.position = uniforms.proj * view_pos;
	output.normal = normal_ws;
	output.tint = input.iTint;
	output.shadow_pos = vec3<f32>(
		shadow_ndc.x * 0.5 + 0.5,
		0.5 - shadow_ndc.y * 0.5,
		shadow_ndc.z,
	);
	return output;
}

@vertex
fn shadow_vertex(input: VertexInput) -> @builtin(position) vec4<f32> {
	let world_pos = world_position(
		input.position,
		input.iPosition,
		input.iRotation,
		input.iScale.xyz,
	);
	return uniforms.light_view_proj * vec4<f32>(world_pos, 1.0);
}

fn srgb_to_linear_channel(c: f32) -> f32 {
	if c <= 0.04045 {
		return c / 12.92;
	}
	return pow((c + 0.055) / 1.055, 2.4);
}

fn unpack_rgba8_srgb(c: u32) -> vec4<f32> {
	let rgba = vec4<f32>(
		f32(c & 255u),
		f32((c >> 8u) & 255u),
		f32((c >> 16u) & 255u),
		f32((c >> 24u) & 255u),
	) / 255.0;

	return vec4<f32>(
		srgb_to_linear_channel(rgba.r),
		srgb_to_linear_channel(rgba.g),
		srgb_to_linear_channel(rgba.b),
		rgba.a,
	);
}

fn sample_shadow(position: vec3<f32>) -> f32 {
	let in_bounds = all(position.xy >= vec2<f32>(0.0)) &&
		all(position.xy <= vec2<f32>(1.0)) &&
		position.z >= 0.0 && position.z <= 1.0;

	if !in_bounds {
		return 1.0;
	}

	return textureSampleCompare(
		shadow_map,
		shadow_sampler,
		position.xy,
		position.z - 0.001,
	);
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
	let color = unpack_rgba8_srgb(input.tint);
	let light = max(dot(input.normal, uniforms.light_dir.xyz), 0.0);
	let shadow = sample_shadow(input.shadow_pos);
	let lighting = mix(0.35, 1.0, light) * mix(0.45, 1.0, shadow);
	return vec4<f32>(color.rgb * lighting, color.a);
}