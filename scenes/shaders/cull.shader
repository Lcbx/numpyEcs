<Frustum> planes: array<vec4f, 6>;

<MeshInstance> iPosition: vec3f;
<MeshInstance> iTint: u32;
<MeshInstance> iRotation: vec2u;
<MeshInstance> iScale: vec2u;

<MeshMetadata> box_center: vec3f;
<MeshMetadata> padding0: f32;
<MeshMetadata> box_extents: vec3f;
<MeshMetadata> padding1: f32;

<CullParams> mesh_id: u32;
<CullParams> instance_offset: u32;
<CullParams> instance_count: u32;
<CullParams> padding: u32;

<DrawIndexedIndirect> index_count: u32;
<DrawIndexedIndirect> instance_count: atomic<u32>;
<DrawIndexedIndirect> first_index: u32;
<DrawIndexedIndirect> base_vertex: i32;
<DrawIndexedIndirect> first_instance: u32;

<CameraUniforms> view_proj: mat4x4f;

@group(0) @binding(0) var<storage, read> instances: array<MeshInstance>;
@group(1) @binding(5) var<uniform> frustum: Frustum;
@group(0) @binding(2) var<storage, read_write> prepass_draw_cmd: DrawIndexedIndirect;
@group(0) @binding(3) var<storage, read_write> frustum_visible_instances: array<u32>;
@group(0) @binding(4) var<storage, read> mesh_metadata: array<MeshMetadata>;
@group(0) @binding(5) var<uniform> cull_params: CullParams;

@group(1) @binding(0) var<uniform> camera_params: CameraUniforms;
@group(1) @binding(1) var hzb_texture: texture_2d<f32>;
@group(1) @binding(3) var<storage, read_write> main_draw_cmd: DrawIndexedIndirect;
@group(1) @binding(4) var<storage, read_write> main_visible_instances: array<u32>;

#include "utils.shaderlib"

@compute @workgroup_size(64)
fn cull_frustum(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let local_id = global_id.x;
	if local_id >= cull_params.instance_count {
		return;
	}

	let instance_id = cull_params.instance_offset + local_id;
	let inst = instances[instance_id];
	let scale = unpack_scale(inst.iScale);
	let rotation = unpack_rotation(inst.iRotation);
	let metadata = mesh_metadata[cull_params.mesh_id];

	let center_ws = inst.iPosition + quat_rotate(rotation, metadata.box_center * scale);
	let axis_x = quat_rotate(rotation, vec3f(metadata.box_extents.x * scale.x, 0.0, 0.0));
	let axis_y = quat_rotate(rotation, vec3f(0.0, metadata.box_extents.y * scale.y, 0.0));
	let axis_z = quat_rotate(rotation, vec3f(0.0, 0.0, metadata.box_extents.z * scale.z));

	var visible = true;
	for (var i = 0u; i < 6u; i = i + 1u) {
		let plane = frustum.planes[i];
		let r = abs(dot(plane.xyz, axis_x)) + abs(dot(plane.xyz, axis_y)) + abs(dot(plane.xyz, axis_z));
		if dot(plane.xyz, center_ws) + plane.w < -r {
			visible = false;
			break;
		}
	}

	if visible {
		let slot = atomicAdd(&prepass_draw_cmd.instance_count, 1u);
		frustum_visible_instances[cull_params.instance_offset + slot] = instance_id;
	}
}

@compute @workgroup_size(64)
fn cull_hiz(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let local_id = global_id.x;
	let frustum_count = atomicLoad(&prepass_draw_cmd.instance_count);
	if local_id >= frustum_count {
		return;
	}

	let instance_id = frustum_visible_instances[cull_params.instance_offset + local_id];
	let inst = instances[instance_id];
	let scale = unpack_scale(inst.iScale);
	let rotation = unpack_rotation(inst.iRotation);
	let metadata = mesh_metadata[cull_params.mesh_id];

	let ext = metadata.box_extents * scale;
	var near_plane = false;
	var min_uv = vec2f(1.0);
	var max_uv = vec2f(0.0);
	var min_z = 1.0;

	for (var i = 0u; i < 8u; i = i + 1u) {
		let corner_local = metadata.box_center * scale + vec3f(
			select(-ext.x, ext.x, (i & 1u) != 0u),
			select(-ext.y, ext.y, (i & 2u) != 0u),
			select(-ext.z, ext.z, (i & 4u) != 0u)
		);
		let corner_ws = inst.iPosition + quat_rotate(rotation, corner_local);
		let clip_pos = camera_params.view_proj * vec4f(corner_ws, 1.0);
		// Keep boxes crossing the near plane; their projected bounds are unstable.
		if clip_pos.w <= 0.0 || clip_pos.z <= 0.0 {
			near_plane = true;
			break;
		}
		let ndc = clip_pos.xyz / clip_pos.w;
		let uv = ndc.xy * vec2f(0.5, -0.5) + vec2f(0.5);

		min_uv = min(min_uv, uv);
		max_uv = max(max_uv, uv);
		min_z = min(min_z, ndc.z);
	}

	var visible = true;
	if !near_plane {
		min_uv = clamp(min_uv, vec2f(0.0), vec2f(1.0));
		max_uv = clamp(max_uv, vec2f(0.0), vec2f(1.0));
		let size = (max_uv - min_uv) * vec2f(textureDimensions(hzb_texture, 0));
		var mip = u32(clamp(ceil(log2(max(max(size.x, size.y), 1.0))), 0.0, f32(textureNumLevels(hzb_texture) - 1u)));
		var mip_size = vec2i(textureDimensions(hzb_texture, mip));
		var lo = clamp(vec2i(floor(min_uv * vec2f(mip_size))), vec2i(0), mip_size - vec2i(1));
		var hi = clamp(vec2i(floor(max_uv * vec2f(mip_size))), vec2i(0), mip_size - vec2i(1));
		// Ensure four loads cover the whole rectangle, including odd-sized mips.
		while any(hi - lo > vec2i(1)) && mip + 1u < textureNumLevels(hzb_texture) {
			mip += 1u;
			mip_size = vec2i(textureDimensions(hzb_texture, mip));
			lo = clamp(vec2i(floor(min_uv * vec2f(mip_size))), vec2i(0), mip_size - vec2i(1));
			hi = clamp(vec2i(floor(max_uv * vec2f(mip_size))), vec2i(0), mip_size - vec2i(1));
		}
		let d0 = textureLoad(hzb_texture, lo, i32(mip)).r;
		let d1 = textureLoad(hzb_texture, vec2i(hi.x, lo.y), i32(mip)).r;
		let d2 = textureLoad(hzb_texture, vec2i(lo.x, hi.y), i32(mip)).r;
		let d3 = textureLoad(hzb_texture, hi, i32(mip)).r;
		let hzb_depth = max(max(d0, d1), max(d2, d3));
		visible = min_z <= hzb_depth + 0.00001;
	}

	if visible {
		let slot = atomicAdd(&main_draw_cmd.instance_count, 1u);
		main_visible_instances[cull_params.instance_offset + slot] = instance_id;
	}
}
