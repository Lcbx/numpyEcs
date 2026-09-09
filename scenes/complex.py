from RenderContext import *
from Utils import *
from ECS import *

from math import cos, sin
import random as rd
import fast_simplification

@component
class Transform:
	position: Vec3
	scale: Vec3
	rotation: Vec4 # Quaternion

@component
class Velocity:
	x: float; y: float; z: float


@component
class MeshRef:
	id: np.uint32
	tint: np.uint32

world = ECS()
transforms, velocities, mesh_refs = world.register(
	Transform, Velocity, MeshRef
)

CUBE_COUNT = 1000
SPACE_SIZE = 180
CUBE_MAX_SIDE = 7


MAX_TRIANGLES = 100
PROBE_SPACING = 0.5
PROBES_HORIZONTAL = 200
PROBE_DIMENSIONS = np.array([PROBES_HORIZONTAL, 8, PROBES_HORIZONTAL], dtype=np.uint32)
PROBE_ORIGIN = Vec3(-PROBES_HORIZONTAL*PROBE_SPACING*0.5, 1.0, -PROBES_HORIZONTAL*PROBE_SPACING*0.5)
PROBE_COUNT = int(np.prod(PROBE_DIMENSIONS))
PROBES_PER_FRAME = (PROBE_COUNT + 15) // 10
BOUNCE_RAYS = 4
PROBE_GEOMETRY_BIAS = 0.08
# 0 direct, 1 bounce, 2 combined, 3 validity, 4 update count
PROBE_DEBUG_MODE = 2

ground = world.create()
world.add(
	ground,
	Transform(
		Vec3(0, -0.51, 0),
		Vec3(2 * SPACE_SIZE, 1, 2 * SPACE_SIZE),
		Quaternion(),
	),
	MeshRef(1, pack_rgba8_srgb([0.5, 0.5, 0.5, 1.0])),
)

cube_entities = world.create(CUBE_COUNT)
cube_count = cube_entities.size

cube_positions = np.column_stack((
	np.random.randint(-SPACE_SIZE, SPACE_SIZE, cube_count),
	np.random.randint(0, 20, cube_count),
	np.random.randint(-SPACE_SIZE, SPACE_SIZE, cube_count),
))
cube_velocities = np.column_stack((
	np.random.randint(-4, 4, cube_count),
	np.zeros(cube_count),
	np.random.randint(-4, 4, cube_count),
))
cube_rotations = np.asarray([
	Quaternion.from_axis_rotation( (0.0,1.0,0.0), rd.random() * 3.143 )
	for _ in range(cube_count)
], dtype=np.float32)

cube_scales = np.column_stack((
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
))

cube_meshrefs = np.column_stack((
	np.full(cube_count, 1, dtype=np.uint32),
	np.asarray([
		pack_rgba8_srgb([rd.random(), rd.random(), rd.random(), 1.0])
		for _ in range(cube_count)
	], dtype=np.uint32)
))

world.add(
	cube_entities,
	Transform, (cube_positions, cube_scales, cube_rotations),
	Velocity, cube_velocities,
	MeshRef, cube_meshrefs,
)

WINDOW_W, WINDOW_H = 1200, 1200
TITLE = "glfw + wgpu"

camera_dist = 30.0
camera = Camera(
	position=(-20.0, 70.0, 25.0),
	target=(0.0, 10.0, 0.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=60.0,
	near=0.1,
	far=1000.0,
)

light_camera = Camera(
	position=(-30.0, 30.0, 25.0),
	target=(0.0, 0.0, -20.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=90.0,
	near=1.0,
	far=300.0,
	perspective=False,
)

RenderContext.init_window(WINDOW_W, WINDOW_H, TITLE, target_fps=-1, required_gpu_features=["indirect-first-instance"])
# RenderContext.capture_mouse()

shader = Shader(filepath='scenes/shaders/complex.shader', label="complex")
main_pipeline = RenderPipeline(
	shader,
	vertex_entry="vertex",
	fragment_entry="fragment",
	label="main",
)
probe_shader = Shader(filepath="scenes/shaders/illumination_probes.shader", label="probes")
probe_pipeline = ComputePipeline(probe_shader, entry="probe_update", label="probe_update")
probe_texture_shader = Shader(filepath="scenes/shaders/probe_textures.shader", label="probe_textures")
probe_texture_pipeline = ComputePipeline(probe_texture_shader, entry="probe_export", label="probe_export")

cull_shader = Shader(filepath='scenes/shaders/cull.shader', label="cull")
cull_pipeline = ComputePipeline(cull_shader, entry="cull", label="cull")

def compute_bounding_box(vertices):
	pos = vertices["position"]
	box_min, box_max = np.min(pos, axis=0), np.max(pos, axis=0)
	return (box_min + box_max) * 0.5, (box_max - box_min) * 0.5

def probe_index(position):
	coord = np.floor((np.asarray(position) - PROBE_ORIGIN) / PROBE_SPACING + 0.5).astype(np.int32)
	if np.any(coord < 0) or np.any(coord >= PROBE_DIMENSIONS): return None
	return int(coord[0] + PROBE_DIMENSIONS[0] * (coord[1] + PROBE_DIMENSIONS[1] * coord[2]))

def probe_position(index):
	x = index % PROBE_DIMENSIONS[0]
	y = (index // PROBE_DIMENSIONS[0]) % PROBE_DIMENSIONS[1]
	z = index // (PROBE_DIMENSIONS[0] * PROBE_DIMENSIONS[1])
	return PROBE_ORIGIN + PROBE_SPACING * np.array([x, y, z], dtype=np.float32)

def selected_probe_ids(first, count):
	return (np.arange(count, dtype=np.uint32) + first) % PROBE_COUNT

def extract_frustum_planes(vp):
	vp = np.asarray(vp)
	planes = np.zeros((6, 4), dtype=np.float32)
	planes[0], planes[1] = vp[:, 3] + vp[:, 0], vp[:, 3] - vp[:, 0]
	planes[2], planes[3] = vp[:, 3] + vp[:, 1], vp[:, 3] - vp[:, 1]
	planes[4], planes[5] = vp[:, 2], vp[:, 3] - vp[:, 2]
	for i in range(6):
		norm = np.linalg.norm(planes[i, :3])
		if norm > 0: planes[i] /= norm
	return planes

vertices, indices = load_gltf_first_mesh_interleaved(
	"scenes/resources/rooftop_utility_pole.glb"
)
model_mesh = Mesh(vertices, indices)

#model_entity = world.create()
#world.add(
#	model_entity,
#	Transform(Vec3(25.0, 1.0, 25.0), Vec3(10.0, 10.0, 10.0), Quaternion()),
#	MeshRef(0, pack_rgba8_srgb([0.3, 0.5, 0.7, 1.0])),
#)

cube_mesh = make_cube_mesh()

meshes = {0: model_mesh, 1: cube_mesh}
draw_batches = []

mesh_metadata_dtype = np.dtype([("box_center", np.float32, 4), ("box_extents", np.float32, 4)])
mesh_metadata = np.zeros(len(meshes), dtype=mesh_metadata_dtype)
mesh_metadata[0]["box_center"][:3], mesh_metadata[0]["box_extents"][:3] = compute_bounding_box(vertices)
mesh_metadata[1]["box_center"][:3], mesh_metadata[1]["box_extents"][:3] = compute_bounding_box(cube_mesh.vertices)
mesh_metadata_buffer = GpuBuffer(mesh_metadata, BufferUsage.STORAGE | BufferUsage.COPY_DST)

frustum_dtype = np.dtype([("planes", np.float32, (6, 4))])
frustum_buffer = cull_shader.UniformBuffer("frustum")


def camera_system(camera, elapsed, camera_dist):
	cam_ang = elapsed * 0.5
	camera.position = Vec3( cos(cam_ang) * camera_dist, camera.position.y, sin(cam_ang) * camera_dist )


def movement_system(world, dt):
	pv = world.where(Transform, Velocity)
	p, v = transforms[pv], velocities[pv]
	p_vec, v_vec = p.position, v.vector()
	p_vec += v_vec * dt

	mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
	mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
	v_vec[mask_x, 0] *= -1
	v_vec[mask_z, 2] *= -1
	p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
	p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE

	p.position = p_vec
	v.set_vector(v_vec)


def render_system(world, instances):

	# NOTE: scale and tint are not modified often, so we could put them in a separated buffer
	# might want to keep data packed in cpu arrays too to avoid needing to pack when uploading the buffer
	# keeping as-is for now
	for mesh, offset, count, sl, entities, indirect_buf, cull_param_buf, cull_bg in draw_batches:
		instances[sl]["iPosition"] = transforms[entities].position
		instances[sl]["iTint"][:, 0] = mesh_refs[entities].tint
		rotations = np.asarray(transforms[entities].rotation, dtype=np.float32)
		lengths = np.linalg.norm(rotations, axis=1, keepdims=True)
		rotations = rotations / np.maximum(lengths, 1e-8)
		rotations[lengths[:, 0] < 1e-8] = [0.0, 0.0, 0.0, 1.0]
		instances[sl]["iRotation"] = pack_quaternion(rotations)
		instances[sl]["iScale"] = pack_scale(transforms[entities].scale)
	
	instance_buffer.content = instances
	instance_buffer.resize(instances.size)

	uniform_buffer.content["view"] = camera.view()
	uniform_buffer.content["proj"] = camera.projection(RenderContext.aspect)
	uniform_buffer.content["light_dir"] = [*light_camera.direction(), 0.0]
	uniform_buffer.content["light_view_proj"] = light_camera.view() @ light_camera.projection(RenderContext.aspect) 

	instance_buffer.upload()
	uniform_buffer.upload()
	update_trace_scene()

	probe_ids = selected_probe_ids(probe_cursor, PROBES_PER_FRAME)
	probe_update_ids.content[:] = probe_ids
	probe_update_ids.upload()
	probe_uniforms.content["update_count"] = probe_ids.size
	probe_uniforms.content["frame_index"] = probe_frame
	probe_uniforms.content["light_direction"] = [*light_camera.direction(), 0.0]
	probe_uniforms.upload()
	with (probe_cmd := RenderContext.commands("probe_update")).compute_pass(label="probe_update") as cp:
		cp.set_pipeline(probe_pipeline)
		cp.set_bind_group(0, probe_update_bindings)
		cp.dispatch((probe_ids.size + 63) // 64)

	# Export all probes so mode changes and invalidated probes are reflected immediately.
	with (probe_texture_cmd := RenderContext.commands("probe_export")).compute_pass(label="probe_export") as cp:
		cp.set_pipeline(probe_texture_pipeline)
		cp.set_bind_group(0, probe_texture_bindings)
		cp.dispatch((PROBE_COUNT + 63) // 64)

	vp = camera.view() @ camera.projection(RenderContext.aspect)
	frustum_buffer.content["planes"] = extract_frustum_planes(vp)
	frustum_buffer.upload()
	clear_cull_cmd = RenderContext.commands("clear_cull")
	with (cull_cmd := RenderContext.commands("cull")).compute_pass(label="cull") as cp:
		cp.set_pipeline(cull_pipeline)
		for mesh, offset, count, sl, entities, indirect_buf, cull_param_buf, cull_bg in draw_batches:
			clear_cull_cmd.clear_buffer(indirect_buf, offset=4, size=4)
			#index_start, _ = mesh.index_range
			#vertex_start, _ = mesh.vertex_range
			#indirect_buf.write(np.array(
			#	[mesh.index_count, 0, index_start, vertex_start, offset],
			#	dtype=np.uint32,
			#))
			cp.set_bind_group(0, cull_bg)
			cp.dispatch((count + 63) // 64)

	with (main_cmd := RenderContext.commands("main")).render_pass(
		color=RenderContext.screen(clear=(0.02, 0.02, 0.03, 1.0)),
		depth=RenderContext.depth_attachment(clear=1.0),
		label="main",
	) as rp:
		rp.set_pipeline(main_pipeline)
		rp.set_bind_group(0, uniform_bindings)
		rp.set_bind_group(1, instance_bindings)
		rp.set_bind_group(2, probe_bindings)
		for mesh, offset, count, sl, entities, indirect_buf, cull_param_buf, cull_bg in draw_batches:
			rp.set_vertex_buffer(0, mesh.vertex_buffer)
			rp.set_index_buffer(mesh.index_buffer, format=mesh.index_format)
			rp.draw_indexed_indirect(indirect_buf)

	RenderContext.submit(
		probe_cmd.finish(),
		probe_texture_cmd.finish(),
		clear_cull_cmd.finish(),
		cull_cmd.finish(),
		main_cmd.finish(),
	)


all_renderables = world.where(Transform, MeshRef)
instances = np.empty(all_renderables.size,
	dtype=mesh_instance_dtype
)

instance_buffer = GpuBuffer(instances, BufferUsage.VERTEX | BufferUsage.STORAGE | BufferUsage.COPY_DST, label='instances')
visible_instances_buffer = GpuBuffer(np.zeros(all_renderables.size, dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST, label='visible_instances')

uniform_buffer = shader.UniformBuffer("uniforms")
uniform_bindings = shader.bind_group(0, uniforms=uniform_buffer)
instance_bindings = shader.bind_group(1,
	instances=instance_buffer,
	visible_instances=visible_instances_buffer
)

probe_dtype = np.dtype([
	("direct", np.float32, (4, 4)),
	("bounce", np.float32, (4, 4)),
	("metadata", np.uint32, 4), # valid, sample count, last update frame, padding
])
trace_instance_dtype = np.dtype([
	("position_radius", np.float32, 4),
	("rotation", np.float32, 4),
	("inverse_scale", np.float32, 4),
	("mesh", np.uint32, 4), # first triangle, triangle count, albedo, padding
])
trace_triangle_dtype = np.dtype([
	("position", np.float32, 4),
	("edge1", np.float32, 4),
	("edge2", np.float32, 4),
])


def build_trace_geometry(meshes, reductions={}):
	triangles = []
	metadata = {}
	offset = 0
	for mesh_id, mesh in meshes.items():
		positions = np.asarray(mesh.vertices["position"], dtype=np.float32)
		start, count = mesh.index_range
		indices = mesh.index_buffer.content[start:start + count]
		if count % 3: raise ValueError("Tracing requires triangle-list meshes")
		if indices.size and np.max(indices) >= len(positions): raise ValueError("Mesh index out of bounds")
		faces = indices.reshape(-1, 3)
		if mesh_id in reductions:
			for _ in range(10):
				old_face_count = len(faces)
				# Weld rendering seams before simplifying the position-only tracing mesh.
				positions, remap = np.unique(positions, axis=0, return_inverse=True)
				positions, faces = fast_simplification.simplify(positions, remap[faces], target_count=MAX_TRIANGLES, agg=10.0)
				positions = np.asarray(positions, dtype=np.float32)
				if not len(faces): raise ValueError("Simplification removed the entire tracing mesh")
				if old_face_count == len(faces): 
					print(f"Trace mesh {mesh_id}: settled at {len(faces)} triangles")
					break
				print(f"Trace mesh {mesh_id}: {old_face_count} -> {len(faces)} triangles")
		vertices = positions[faces]
		data = np.zeros(len(vertices), dtype=trace_triangle_dtype)
		data["position"][:, :3] = vertices[:, 0]
		data["edge1"][:, :3] = vertices[:, 1] - vertices[:, 0]
		data["edge2"][:, :3] = vertices[:, 2] - vertices[:, 0]
		# Origin-centered bound: no rotation-dependent CPU work.
		radius = float(np.max(np.linalg.norm(positions, axis=1))) if len(positions) else 0.0
		metadata[mesh_id] = (offset, len(data), radius)
		triangles.append(data)
		offset += len(data)
	# Keep an empty scene's storage binding nonempty.
	return np.concatenate(triangles) if offset else np.zeros(1, dtype=trace_triangle_dtype), metadata


trace_triangles, trace_metadata = build_trace_geometry(meshes, {0,})
trace_triangle_buffer = GpuBuffer(trace_triangles, BufferUsage.STORAGE, label="trace_triangles")
probe_storage = np.zeros(PROBE_COUNT, dtype=probe_dtype)
probe_buffer = GpuBuffer(probe_storage, BufferUsage.STORAGE | BufferUsage.COPY_DST, label="probes")
probe_update_ids = GpuBuffer(np.zeros(PROBES_PER_FRAME, dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST, label="probe_update_ids")
trace_instances = np.zeros(all_renderables.size, dtype=trace_instance_dtype)
trace_instance_buffer = GpuBuffer(trace_instances, BufferUsage.STORAGE | BufferUsage.COPY_DST, label="trace_instances")
probe_uniforms = probe_shader.UniformBuffer("probe_uniforms")
probe_uniforms.content["origin"] = [*PROBE_ORIGIN, PROBE_SPACING]
probe_uniforms.content["dimensions"] = [*PROBE_DIMENSIONS, PROBE_COUNT]
probe_uniforms.content["light_direction"] = [*light_camera.direction(), 0.0]
probe_uniforms.content["light_radiance"] = [4.0, 3.8, 3.5, 0.0]
probe_uniforms.content["trace"] = [all_renderables.size, BOUNCE_RAYS, 0, PROBE_DEBUG_MODE]
probe_uniforms.content["geometry_bias"] = PROBE_GEOMETRY_BIAS
probe_update_bindings = probe_shader.bind_group(0,
	probe_uniforms=probe_uniforms,
	probes=probe_buffer,
	probe_update_ids=probe_update_ids,
	trace_instances=trace_instance_buffer,
	trace_triangles=trace_triangle_buffer,
)
# Four coefficient volumes: RGB = selected irradiance SH, alpha = validity.
# Accumulation stays in the float32 buffer; the render volumes use float16.
probe_textures = [Texture(
	tuple(int(value) for value in PROBE_DIMENSIONS),
	format="rgba16float",
	usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
	dimension="3d",
	label=f"probe_sh{band}",
) for band in range(4)]
probe_sampler = Sampler(min_filter="linear", mag_filter="linear", address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge", address_mode_w="clamp-to-edge")
probe_texture_bindings = probe_texture_shader.bind_group(0,
	probe_uniforms=probe_uniforms,
	probes=probe_buffer,
	**{f"probe_sh{band}": texture for band, texture in enumerate(probe_textures)},
)
probe_bindings = shader.bind_group(2,
	probe_uniforms=probe_uniforms,
	probe_sampler=probe_sampler,
	**{f"probe_sh{band}": texture for band, texture in enumerate(probe_textures)},
)

trace_batches = []
for mesh_id in meshes:
	rows = np.flatnonzero(mesh_refs[all_renderables].id == mesh_id)
	trace_batches.append((mesh_id, rows, all_renderables[rows]))


def update_trace_scene():
	for mesh_id, rows, entities in trace_batches:
		first, count, radius = trace_metadata[mesh_id]
		positions = transforms[entities].position
		scales = transforms[entities].scale
		rotations = np.asarray(transforms[entities].rotation, dtype=np.float32)
		lengths = np.linalg.norm(rotations, axis=1, keepdims=True)
		rotations = rotations / np.maximum(lengths, 1e-8)
		rotations[lengths[:, 0] < 1e-8] = [0.0, 0.0, 0.0, 1.0]
		valid = np.all(np.abs(scales) >= 1e-8, axis=1)
		trace_instances["position_radius"][rows, :3] = positions
		trace_instances["position_radius"][rows, 3] = radius * np.max(np.abs(scales), axis=1) * 1.00001
		trace_instances["rotation"][rows] = rotations
		inverse_scale = np.zeros_like(scales)
		np.divide(1.0, scales, out=inverse_scale, where=np.abs(scales) >= 1e-8)
		trace_instances["inverse_scale"][rows, :3] = inverse_scale
		trace_instances["mesh"][rows, 0] = first
		trace_instances["mesh"][rows, 1] = np.where(valid, count, 0)
		trace_instances["mesh"][rows, 2] = mesh_refs[entities].tint
	trace_instance_buffer.content = trace_instances
	trace_instance_buffer.upload()


offset = 0
for mesh_id, mesh in meshes.items():
	entities = all_renderables[mesh_refs[all_renderables].id == mesh_id]
	count = entities.size
	if count == 0: continue
	sl = slice(offset, offset + count)

	index_start,  _ = mesh.index_range
	vertex_start, _ = mesh.vertex_range
	indirect_data = np.array(
		[mesh.index_count, 0, index_start, vertex_start, offset],
		dtype=np.uint32,
	)
	indirect_buf = GpuBuffer(indirect_data, BufferUsage.STORAGE  | BufferUsage.INDIRECT | BufferUsage.COPY_DST)
	
	cull_param_data = np.array([mesh_id, offset, count, 0], dtype=np.uint32)
	cull_param_buf = GpuBuffer(cull_param_data, BufferUsage.UNIFORM | BufferUsage.COPY_DST)
	
	cull_bg = cull_shader.bind_group(
		0,
		instances=instance_buffer,
		frustum=frustum_buffer,
		draw_cmd=indirect_buf,
		visible_instances=visible_instances_buffer,
		mesh_metadata=mesh_metadata_buffer,
		cull_params=cull_param_buf,
	)
	
	draw_batches.append((mesh, offset, count, sl, entities, indirect_buf, cull_param_buf, cull_bg))
	offset += count
	# initialisation
	instances[sl]["iPosition"] = transforms[entities].position
	instances[sl]["iTint"][:, 0] = mesh_refs[entities].tint
	instances[sl]["iRotation"] = pack_quaternion(transforms[entities].rotation)
	instances[sl]["iScale"] = pack_scale(transforms[entities].scale)

def clamp(val, val_min, val_max):
	return min(max(val, val_min), val_max)


def scroll_callback(xoff, yoff):
	global camera_dist, camera
	camera_dist = clamp(camera_dist - 5.0 * yoff, 5.0, 100.0)
	cp = camera.position
	y_factor = 1.0 if cp.y < 15.0 else 3.0
	camera.position = Vec3(cp.x, cp.y - y_factor * yoff, cp.z)
	return True


RenderContext.event_handlers["mouse_scroll"].append(scroll_callback)

fps_frames = 0
start_t = get_time()
fps_print_timestamp = start_t
probe_cursor = 0
probe_frame = 0

while RenderContext.window_loop():
	now = RenderContext.frame_start

	fps_frames += 1
	if now - fps_print_timestamp >= 1.0:
		print(f"fps {fps_frames}")
		fps_frames = 0
		fps_print_timestamp = now

	elapsed = now - start_t
	camera_system(camera, elapsed, camera_dist)
	movement_system(world, RenderContext.frame_time)
	render_system(world, instances)
	probe_cursor = (probe_cursor + PROBES_PER_FRAME) % PROBE_COUNT
	probe_frame += 1
