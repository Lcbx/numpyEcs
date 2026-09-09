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

@group(0) @binding(0) var<uniform> probe_uniforms: ProbeUniforms;
@group(0) @binding(1) var<storage, read> probes: array<Probe>;
@group(0) @binding(2) var probe_sh0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var probe_sh1: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var probe_sh2: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var probe_sh3: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(64)
fn probe_export(@builtin(global_invocation_id) gid: vec3u) {
	let id = gid.x;
	if id >= probe_uniforms.dimensions.w { return; }
	let dims = probe_uniforms.dimensions.xyz;
	let coord = vec3i(i32(id % dims.x), i32((id / dims.x) % dims.y), i32(id / (dims.x * dims.y)));
	let probe = probes[id];
	let valid = probe.metadata.x != 0u && probe.metadata.y != 0u;
	var coefficients = array<vec4f, 4>();
	if valid {
		for (var band = 0u; band < 4u; band++) {
			coefficients[band] = vec4f(clamp(probe.bounce[band].rgb, vec3f(-65504.0), vec3f(65504.0)), 0.0);
		}
		// All channels are validity-weighted; invalid probes export zero everywhere.
		coefficients[0].a = 1.0;
		coefficients[1].a = clamp(probe.visibility.x, 0.0, 1.0);
		coefficients[2].a = min(log2(f32(probe.metadata.y) + 1.0) / 8.0, 1.0);
	}

	textureStore(probe_sh0, coord, coefficients[0]);
	textureStore(probe_sh1, coord, coefficients[1]);
	textureStore(probe_sh2, coord, coefficients[2]);
	textureStore(probe_sh3, coord, coefficients[3]);
}
