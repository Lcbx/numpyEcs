uniform view: mat4x4<f32>;
uniform proj: mat4x4<f32>;
uniform light_dir: vec4<f32>;
uniform light_view_proj: mat4x4<f32>;

texture shadow_map: texture_depth_2d;
sampler shadow_sampler: sampler_comparison;

in position: vec3<f32>;
in normal: vec3<f32>;
in uv: vec2<f32>;
in iPosition: vec3<f32>;
in iRotation: vec4<f32>;
in iScale: vec4<f32>;
in iTint: u32;

@position
varying position: vec4<f32>;
varying normal: vec3<f32>;
flat varying tint: u32;
varying shadow_pos: vec3<f32>;

out color: vec4<f32>;


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
fn vertex() {
	let world_pos = world_position(
		in.position,
		in.iPosition,
		in.iRotation,
		in.iScale.xyz,
	);

	var normal = in.normal;
	if any(in.iScale.xyz != vec3<f32>(1.0)) {
		normal /= in.iScale.xyz;
	}

	let normal_ws = normalize(quat_rotate(in.iRotation, normal));
	let world = vec4<f32>(world_pos, 1.0);
	let view_pos = view * world;
	let shadow_clip = light_view_proj * world;
	let shadow_ndc = shadow_clip.xyz / shadow_clip.w;

	out.position = proj * view_pos;
	out.normal = normal_ws;
	out.tint = in.iTint;
	out.shadow_pos = vec3<f32>(
		shadow_ndc.x * 0.5 + 0.5,
		0.5 - shadow_ndc.y * 0.5,
		shadow_ndc.z,
	);
}

@vertex
fn shadow_vertex() {
	let world_pos = world_position(
		in.position,
		in.iPosition,
		in.iRotation,
		in.iScale.xyz,
	);

	out.position = light_view_proj * vec4<f32>(world_pos, 1.0);
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
fn fragment() {
	let color = unpack_rgba8_srgb(in.tint);
	let light = max(dot(in.normal, light_dir.xyz), 0.0);
	let shadow = sample_shadow(in.shadow_pos);
	let lighting = mix(0.35, 1.0, light) * mix(0.45, 1.0, shadow);

	out.color = vec4<f32>(color.rgb * lighting, color.a);
}