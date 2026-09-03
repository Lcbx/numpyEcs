
<Uniforms> view: mat4x4f;
<Uniforms> proj: mat4x4f;
<Uniforms> light_dir: vec4f;
<Uniforms> light_view_proj: mat4x4f;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var shadow_map: texture_depth_2d;

@group(1) @binding(1)
var shadow_sampler: sampler_comparison;

<VertexInput> position: vec3f;
<VertexInput> normal: vec3f;
<VertexInput> uv: vec2f;
<VertexInput> iPosition: vec3f;
<VertexInput> iTint: u32;
<VertexInput> iRotation: vec2u;
<VertexInput> iScale: vec2u;

<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> normal: vec3f;
<VertexOutput> @interpolate(flat) tint: u32;
<VertexOutput> shadow_pos: vec3f;

fn unpack_rotation(packed: vec2u) -> vec4f {
    return vec4f(
        unpack2x16snorm(packed.x),
        unpack2x16snorm(packed.y)
    );
}

fn unpack_scale(packed: vec2u) -> vec3f {
    let xy = unpack2x16float(packed.x);
    let z  = unpack2x16float(packed.y).x;
    return vec3f(xy, z);
}

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
	let t = cross(q.xyz, v) * 2.0;
	return v + q.w * t + cross(q.xyz, t);
}

fn world_position(
	position: vec3f,
	iPosition: vec3f,
	iRotation: vec4f,
	iScale: vec3f,
) -> vec3f {
	return iPosition + quat_rotate(iRotation, position * iScale);
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {

	let rotation  = unpack_rotation(input.iRotation);
	let scale     = unpack_scale(input.iScale);

	let world_pos = world_position(
		input.position,
		input.iPosition,
		rotation,
		scale,
	);

	var normal = input.normal;
	if any(scale != vec3f(1.0)) {
		normal /= scale;
	}

	let normal_ws = normalize(quat_rotate(rotation, normal));
	let world = vec4f(world_pos, 1.0);
	let view_pos = uniforms.view * world;
	let shadow_clip = uniforms.light_view_proj * world;
	let shadow_ndc = shadow_clip.xyz / shadow_clip.w;

	var output: VertexOutput;
	output.position = uniforms.proj * view_pos;
	output.normal = normal_ws;
	output.tint = input.iTint;
	output.shadow_pos = vec3f(
		shadow_ndc.x * 0.5 + 0.5,
		0.5 - shadow_ndc.y * 0.5,
		shadow_ndc.z,
	);
	return output;
}

@vertex
fn shadow_vertex(input: VertexInput) -> @builtin(position) vec4f {

	let world_pos = world_position(
		input.position,
		input.iPosition,
		unpack_rotation(input.iRotation),
		unpack_scale(input.iScale),
	);
	return uniforms.light_view_proj * vec4f(world_pos, 1.0);
}

fn srgb_to_linear_channel(c: f32) -> f32 {
	if c <= 0.04045 {
		return c / 12.92;
	}
	return pow((c + 0.055) / 1.055, 2.4);
}

fn unpack_rgba8_srgb(c: u32) -> vec4f {
	let rgba = vec4f(
		f32(c & 255u),
		f32((c >> 8u) & 255u),
		f32((c >> 16u) & 255u),
		f32((c >> 24u) & 255u),
	) / 255.0;

	return vec4f(
		srgb_to_linear_channel(rgba.r),
		srgb_to_linear_channel(rgba.g),
		srgb_to_linear_channel(rgba.b),
		rgba.a,
	);
}

fn sample_shadow(position: vec3f) -> f32 {
	let in_bounds = all(position.xy >= vec2f(0.0)) &&
		all(position.xy <= vec2f(1.0)) &&
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
fn fragment(input: VertexOutput) -> @location(0) vec4f {
	let color = unpack_rgba8_srgb(input.tint);
	let light = max(dot(input.normal, uniforms.light_dir.xyz), 0.0);
	let shadow = sample_shadow(input.shadow_pos);
	let lighting = mix(0.35, 1.0, light) * mix(0.45, 1.0, shadow);
	return vec4f(color.rgb * lighting, color.a);
}