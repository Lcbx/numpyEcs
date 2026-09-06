
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


@group(0) @binding(0) var<storage, read> instances: array<MeshInstance>;
@group(0) @binding(1) var<uniform> frustum: Frustum;
@group(0) @binding(2) var<storage, read_write> draw_cmd: DrawIndexedIndirect;
@group(0) @binding(3) var<storage, read_write> visible_instances: array<u32>;
@group(0) @binding(4) var<storage, read> mesh_metadata: array<MeshMetadata>;
@group(0) @binding(5) var<uniform> cull_params: CullParams;


fn unpack_rotation(packed: vec2u) -> vec4f {
	return vec4f( unpack2x16snorm(packed.x), unpack2x16snorm(packed.y) );
}

fn unpack_scale(packed: vec2u) -> vec3f {
	return vec3f( unpack2x16float(packed.x).xy, unpack2x16float(packed.y).x );
}


fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
	let t = cross(q.xyz, v) * 2.0;
	return v + q.w * t + cross(q.xyz, t);
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
		let slot = atomicAdd(&draw_cmd.instance_count, 1u);
		visible_instances[cull_params.instance_offset + slot] = instance_id;
	}
}