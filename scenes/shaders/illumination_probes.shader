

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

<TraceInstance> box_min: vec4f;
<TraceInstance> box_max: vec4f;
<TraceInstance> albedo:  u32;

@group(0) @binding(0) var<uniform> probe_uniforms: ProbeUniforms;
@group(0) @binding(1) var<storage, read_write> probes: array<Probe>;
@group(0) @binding(2) var<storage, read> probe_update_ids: array<u32>;
@group(0) @binding(3) var<storage, read> trace_instances: array<TraceInstance>;

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

fn ray_box(origin: vec3f, direction: vec3f, box_min: vec3f, box_max: vec3f) -> f32 {
	let inv = 1.0 / direction;
	let t0 = (box_min - origin) * inv;
	let t1 = (box_max - origin) * inv;
	let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
	let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
	return select(1e30, max(near, 0.0), far >= max(near, 0.0));
}

fn trace_scene(origin: vec3f, direction: vec3f) -> vec4f {
	var best = 1e30;
	var hit_id = 0xffffffffu;
	for (var i = 0u; i < probe_uniforms.trace.x; i++) {
		let hit = ray_box(origin, direction, trace_instances[i].box_min.xyz, trace_instances[i].box_max.xyz);
		if hit < best { best = hit; hit_id = i; }
	}
	return vec4f(best, bitcast<f32>(hit_id), 0.0, 0.0);
}

fn hit_normal(position: vec3f, instance: TraceInstance) -> vec3f {
	let center = (instance.box_min.xyz + instance.box_max.xyz) * 0.5;
	let extents = max((instance.box_max.xyz - instance.box_min.xyz) * 0.5, vec3f(1e-5));
	let local = (position - center) / extents;
	let a = abs(local);
	if a.x > a.y && a.x > a.z { return vec3f(sign(local.x), 0.0, 0.0); }
	if a.y > a.z { return vec3f(0.0, sign(local.y), 0.0); }
	return vec3f(0.0, 0.0, sign(local.z));
}

fn fibonacci_direction(sample: u32, count: u32, seed: u32) -> vec3f {
	let z = 1.0 - 2.0 * (f32(sample) + 0.5) / f32(count);
	let angle = 2.39996323 * f32(sample) + 6.28318531 * fract(sin(f32(seed) * 12.9898) * 43758.5453);
	let radius = sqrt(max(0.0, 1.0 - z * z));
	return vec3f(cos(angle) * radius, z, sin(angle) * radius);
}

fn visible_to_light(position: vec3f, normal: vec3f) -> bool {
	let origin = position + normal * 0.08;
	return trace_scene(origin, normalize(probe_uniforms.light_direction.xyz)).x == 1e30;
}

@compute @workgroup_size(64)
fn probe_update(@builtin(global_invocation_id) gid: vec3u) {
	if gid.x >= probe_uniforms.update_count { return; }
	let id = probe_update_ids[gid.x];
	let position = probe_world_position(id);
	var direct = array<vec4f, 4>();
	var bounce = array<vec4f, 4>();
	var valid = true;

	for (var i = 0u; i < probe_uniforms.trace.x; i++) {
		let box = trace_instances[i];
		if all(position >= box.box_min.xyz - vec3f(0.08)) && all(position <= box.box_max.xyz + vec3f(0.08)) { valid = false; }
	}
	if !valid {
		probes[id].metadata = vec4u(0u);
		return;
	}

	let light_dir = normalize(probe_uniforms.light_direction.xyz);
	if visible_to_light(position, light_dir) {
		let basis = sh_basis(light_dir);
		for (var band = 0u; band < 4u; band++) {
			let convolution = select(2.09439510, 3.14159265, band == 0u);
			direct[band] = vec4f(probe_uniforms.light_radiance.rgb * basis[band] * convolution, 0.0);
		}
	}

	let ray_count = max(probe_uniforms.trace.y, 1u);
	for (var sample = 0u; sample < ray_count; sample++) {
		let direction = fibonacci_direction(sample, ray_count, id + probe_uniforms.frame_index * 1664525u);
		let hit = trace_scene(position, direction);
		if hit.x == 1e30 { continue; }
		let hit_id = bitcast<u32>(hit.y);
		let surface = position + direction * hit.x;
		let normal = hit_normal(surface, trace_instances[hit_id]);
		let n_dot_l = max(dot(normal, light_dir), 0.0);
		if n_dot_l == 0.0 || !visible_to_light(surface, normal) { continue; }
		let radiance = unpack_rgba8_srgb(trace_instances[hit_id].albedo).rgb * probe_uniforms.light_radiance.rgb * n_dot_l * (1.0 / 3.14159265);
		let basis = sh_basis(direction);
		for (var band = 0u; band < 4u; band++) {
			let convolution = select(2.09439510, 3.14159265, band == 0u);
			bounce[band] += vec4f(radiance * basis[band] * convolution * (12.56637061 / f32(ray_count)), 0.0);
		}
	}

	let old_count = probes[id].metadata.y;
	let alpha = select(0.08, 1.0 / f32(old_count + 1u), old_count < 8u);
	for (var band = 0u; band < 4u; band++) {
		probes[id].direct[band] = mix(probes[id].direct[band], direct[band], alpha);
		probes[id].bounce[band] = mix(probes[id].bounce[band], bounce[band], alpha);
	}
	probes[id].metadata = vec4u(1u, old_count + 1u, probe_uniforms.frame_index, 0u);
}
