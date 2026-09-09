

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

<Probe> visibility: vec4f;
<Probe> bounce:   array<vec4f, 4>;
<Probe> metadata: vec4u;

<TraceInstance> position_radius: vec4f;
<TraceInstance> rotation: vec4f;
<TraceInstance> inverse_scale: vec4f;
<TraceInstance> mesh: vec4u;

<TraceTriangle> position: vec4f;
<TraceTriangle> edge1: vec4f;
<TraceTriangle> edge2: vec4f;

<TraceHit> distance: f32;
<TraceHit> instance_id: u32;
<TraceHit> normal: vec3f;

@group(0) @binding(0) var<uniform> probe_uniforms: ProbeUniforms;
@group(0) @binding(1) var<storage, read_write> probes: array<Probe>;
@group(0) @binding(2) var<storage, read> probe_update_ids: array<u32>;
@group(0) @binding(3) var<storage, read> trace_instances: array<TraceInstance>;
@group(0) @binding(4) var<storage, read> trace_triangles: array<TraceTriangle>;

const SH0 = 0.28209479;
const SH1 = 0.48860251;

fn sh_basis(direction: vec3f) -> vec4f {
	return vec4f(SH0, SH1 * direction.y, SH1 * direction.z, SH1 * direction.x);
}

#include "utils.shaderlib"
#from "utils.shaderlib" import instance_transform_unpacking, unpack_srgb_color
{{ unpack_srgb_color() }}


fn probe_world_position(id: u32) -> vec3f {
	let dims = probe_uniforms.dimensions.xyz;
	let x = id % dims.x;
	let y = (id / dims.x) % dims.y;
	let z = id / (dims.x * dims.y);
	return probe_uniforms.origin.xyz + vec3f(f32(x), f32(y), f32(z)) * probe_uniforms.origin.w;
}

fn trace_rotate(rotation: vec4f, value: vec3f) -> vec3f {
	return value + 2.0 * cross(rotation.xyz, cross(rotation.xyz, value) + rotation.w * value);
}

fn ray_sphere(origin: vec3f, direction: vec3f, sphere: vec4f, limit: f32) -> bool {
	let offset = origin - sphere.xyz;
	let a = dot(direction, direction);
	let b = dot(offset, direction);
	let c = dot(offset, offset) - sphere.w * sphere.w;
	if c <= 0.0 { return true; }
	if b > 0.0 { return false; }
	let discriminant = b * b - a * c;
	if discriminant < 0.0 { return false; }
	return (-b - sqrt(discriminant)) <= limit * a;
}

fn ray_triangle(origin: vec3f, direction: vec3f, triangle: TraceTriangle) -> f32 {
	let p = cross(direction, triangle.edge2.xyz);
	let determinant = dot(triangle.edge1.xyz, p);
	let tolerance = 1e-7 * length(triangle.edge1.xyz) * length(p);
	if abs(determinant) <= tolerance { return 1e30; }
	let offset = origin - triangle.position.xyz;
	let u = dot(offset, p) / determinant;
	if u < 0.0 || u > 1.0 { return 1e30; }
	let q = cross(offset, triangle.edge1.xyz);
	let v = dot(direction, q) / determinant;
	if v < 0.0 || u + v > 1.0 { return 1e30; }
	let distance = dot(triangle.edge2.xyz, q) / determinant;
	return select(1e30, distance, distance >= 0.0);
}

// Local directions are not normalized: the ray parameter stays in world units.
fn trace_scene(origin: vec3f, direction: vec3f, any_hit: bool) -> TraceHit {
	var result = TraceHit(1e30, 0xffffffffu, vec3f(0.0));
	for (var i = 0u; i < probe_uniforms.trace.x; i++) {
		let instance = trace_instances[i];
		if instance.mesh.y == 0u || !ray_sphere(origin, direction, instance.position_radius, result.distance) { continue; }
		let inverse_rotation = vec4f(-instance.rotation.xyz, instance.rotation.w);
		let local_origin = trace_rotate(inverse_rotation, origin - instance.position_radius.xyz) * instance.inverse_scale.xyz;
		let local_direction = trace_rotate(inverse_rotation, direction) * instance.inverse_scale.xyz;
		for (var j = 0u; j < instance.mesh.y; j++) {
			let triangle = trace_triangles[instance.mesh.x + j];
			let distance = ray_triangle(local_origin, local_direction, triangle);
			if distance >= result.distance { continue; }
			result.distance = distance;
			result.instance_id = i;
			if any_hit { return result; }
			let normal = cross(triangle.edge1.xyz, triangle.edge2.xyz) * instance.inverse_scale.xyz;
			result.normal = normalize(trace_rotate(instance.rotation, normal));
			// Treat surfaces as two-sided for diffuse transport and occlusion.
			if dot(result.normal, direction) > 0.0 { result.normal = -result.normal; }
		}
	}
	return result;
}

fn segment_distance_squared(point: vec3f, a: vec3f, b: vec3f) -> f32 {
	let edge = b - a;
	let t = clamp(dot(point - a, edge) / max(dot(edge, edge), 1e-30), 0.0, 1.0);
	let delta = point - (a + edge * t);
	return dot(delta, delta);
}

fn triangle_distance_squared(point: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
	let normal = cross(b - a, c - a);
	let normal_squared = dot(normal, normal);
	if normal_squared > 1e-30 {
		if dot(cross(b - a, point - a), normal) >= 0.0 &&
			dot(cross(c - b, point - b), normal) >= 0.0 &&
			dot(cross(a - c, point - c), normal) >= 0.0 {
			let height = dot(point - a, normal);
			return height * height / normal_squared;
		}
	}
	return min(segment_distance_squared(point, a, b), min(segment_distance_squared(point, b, c), segment_distance_squared(point, c, a)));
}

fn probe_is_valid(position: vec3f) -> bool {
	let bias = probe_uniforms.geometry_bias;
	for (var i = 0u; i < probe_uniforms.trace.x; i++) {
		let instance = trace_instances[i];
		if instance.mesh.y == 0u { continue; }
		let delta = position - instance.position_radius.xyz;
		let radius = instance.position_radius.w + bias;
		if dot(delta, delta) > radius * radius { continue; }
		let inverse_rotation = vec4f(-instance.rotation.xyz, instance.rotation.w);
		// Undo rotation only so proximity distances still use world units.
		let point = trace_rotate(inverse_rotation, delta);
		let scale = 1.0 / instance.inverse_scale.xyz;
		var winding = 0.0;
		for (var j = 0u; j < instance.mesh.y; j++) {
			let triangle = trace_triangles[instance.mesh.x + j];
			let a = triangle.position.xyz * scale;
			let b = (triangle.position.xyz + triangle.edge1.xyz) * scale;
			let c = (triangle.position.xyz + triangle.edge2.xyz) * scale;
			if triangle_distance_squared(point, a, b, c) <= bias * bias { return false; }
			let u = a - point;
			let v = b - point;
			let w = c - point;
			let numerator = dot(u, cross(v, w));
			if numerator == 0.0 { continue; }
			let denominator = length(u) * length(v) * length(w) + dot(u, v) * length(w) + dot(v, w) * length(u) + dot(w, u) * length(v);
			winding += 2.0 * atan2(numerator, denominator);
		}
		// Closed, consistently wound meshes have |winding| = 4*pi inside.
		// For open meshes this is only a heuristic; surface proximity still applies.
		if abs(winding) > 6.28318531 { return false; }
	}
	return true;
}

fn fibonacci_direction(sample: u32, count: u32, seed: u32) -> vec3f {
	let z = 1.0 - 2.0 * (f32(sample) + 0.5) / f32(count);
	let angle = 2.39996323 * f32(sample) + 6.28318531 * fract(sin(f32(seed) * 12.9898) * 43758.5453);
	let radius = sqrt(max(0.0, 1.0 - z * z));
	return vec3f(cos(angle) * radius, z, sin(angle) * radius);
}


fn light_random(seed: u32) -> f32 {
	var value = seed;
	value = (value ^ (value >> 16u)) * 0x7feb352du;
	value = (value ^ (value >> 15u)) * 0x846ca68bu;
	value = value ^ (value >> 16u);
	return f32(value >> 8u) * (1.0 / 16777216.0);
}

// Uniform solid-angle sampling, stratified along the cone's cosine coordinate.
fn sample_light_direction(light_dir: vec3f, sample: u32, count: u32, seed: u32) -> vec3f {
	let u = (f32(sample) + light_random(seed + sample * 0x9e3779b9u)) / f32(count);
	let v = light_random(seed + sample * 0x85ebca6bu + 0x68bc21ebu);
	let cosine = mix(1.0, probe_uniforms.light_sampling.x, u);
	let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
	let axis = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(light_dir.y) > 0.9);
	let tangent = normalize(cross(axis, light_dir));
	let bitangent = cross(light_dir, tangent);
	let angle = 6.28318531 * v;
	return normalize(light_dir * cosine + (tangent * cos(angle) + bitangent * sin(angle)) * sine);
}

fn visible_to_light(position: vec3f, normal: vec3f, direction: vec3f) -> bool {
	let origin = position + normal * probe_uniforms.geometry_bias;
	return trace_scene(origin, direction, true).instance_id == 0xffffffffu;
}

@compute @workgroup_size(64)
fn probe_update(@builtin(global_invocation_id) gid: vec3u) {
	if gid.x >= probe_uniforms.update_count { return; }
	let id = probe_update_ids[gid.x];
	let position = probe_world_position(id);
	var bounce = array<vec4f, 4>();
	if !probe_is_valid(position) {
		probes[id].metadata = vec4u(0u);
		return;
	}

	let light_dir = normalize(probe_uniforms.light_direction.xyz);
	let light_seed = id * 0x9e3779b9u + probe_uniforms.frame_index * 0x85ebca6bu;
	let direct_count = max(probe_uniforms.trace.z, 1u);
	var visibility = 0.0;
	for (var sample = 0u; sample < direct_count; sample++) {
		let direction = sample_light_direction(light_dir, sample, direct_count, light_seed);
		visibility += select(0.0, 1.0, visible_to_light(position, direction, direction));
	}
	visibility /= f32(direct_count);

	let ray_count = max(probe_uniforms.trace.y, 1u);
	for (var sample = 0u; sample < ray_count; sample++) {
		let direction = fibonacci_direction(sample, ray_count, id + probe_uniforms.frame_index * 1664525u);
		let hit = trace_scene(position, direction, false);
		if hit.distance == 1e30 { continue; }
		let hit_id = hit.instance_id;
		let surface = position + direction * hit.distance;
		let normal = hit.normal;
		
		// One light sample per bounce hit keeps the same finite source without extra shadow rays.
		let bounce_light_dir = sample_light_direction(light_dir, 0u, 1u, light_seed + sample * 0xc2b2ae35u + 0x27d4eb2fu);
		let n_dot_l = max(dot(normal, bounce_light_dir), 0.0);
		if n_dot_l == 0.0 || !visible_to_light(surface, normal, bounce_light_dir) { continue; }
		let radiance = unpack_rgba8_srgb(trace_instances[hit_id].mesh.z).rgb * probe_uniforms.light_radiance.rgb * n_dot_l * (1.0 / 3.14159265);
		let basis = sh_basis(direction);
		for (var band = 0u; band < 4u; band++) {
			let convolution = select(2.09439510, 3.14159265, band == 0u);
			bounce[band] += vec4f(radiance * basis[band] * convolution * (12.56637061 / f32(ray_count)), 0.0);
		}
	}

	let old_count = probes[id].metadata.y;
	let alpha = select(0.08, 1.0 / f32(old_count + 1u), old_count < 8u);
	for (var band = 0u; band < 4u; band++) {
		probes[id].bounce[band] = mix(probes[id].bounce[band], bounce[band], alpha);
	}
	let visibility_alpha = max(clamp(probe_uniforms.light_sampling.y, 0.0, 1.0), 1.0 / f32(old_count + 1u));
	probes[id].visibility = vec4f(mix(probes[id].visibility.x, visibility, visibility_alpha), 0.0, 0.0, 0.0);
	probes[id].metadata = vec4u(1u, old_count + 1u, probe_uniforms.frame_index, 0u);
}
