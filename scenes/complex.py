from RenderContext import *
from Utils import *
from ECS import *

from math import cos, sin
import random as rd


@component
class Position:
	x: float; y: float; z: float


@component
class Velocity:
	x: float; y: float; z: float


@component
class Rotation:
	x: float; y: float; z: float; w: float


@component
class Scale:
	x: float; y: float; z: float; w: float # w is not used


@component
class MeshRef:
	id: np.uint32
	tint: np.uint32


world = ECS()
positions, velocities, rotations, scales, mesh_refs = world.register(
	Position, Velocity, Rotation, Scale, MeshRef
)

CUBE_COUNT = 1000
SPACE_SIZE = 180
CUBE_MAX_SIDE = 7

ground = world.create()
world.add(
	ground,
	Position, Position(0, -0.51, 0),
	Rotation, Rotation(*Quaternion()),
	Scale, Scale(2 * SPACE_SIZE, 1, 2 * SPACE_SIZE, 0),
	MeshRef, MeshRef(1, pack_rgba8_srgb([0.5, 0.5, 0.5, 1.0])),
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
cube_rotations = np.tile(np.asarray(Quaternion()), (cube_count, 1))
cube_scales = np.column_stack((
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
	np.zeros(cube_count),
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
	Position, cube_positions,
	Velocity, cube_velocities,
	Rotation, cube_rotations,
	Scale, cube_scales,
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
shadow_pipeline = RenderPipeline(
	shader,
	vertex_entry="shadow_vertex",
	depth_bias=2,
	depth_bias_slope_scale=2.0,
	label="shadow",
)

cull_shader = Shader(filepath='scenes/shaders/cull.shader', label="cull")
cull_pipeline = ComputePipeline(cull_shader, entry="cull", label="cull")

def compute_bounding_box(vertices):
	pos = vertices["position"]
	box_min, box_max = np.min(pos, axis=0), np.max(pos, axis=0)
	return (box_min + box_max) * 0.5, (box_max - box_min) * 0.5

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

model_entity = world.create()
world.add(
	model_entity,
	Position, Position(15.0, 0.0, 15.0),
	Rotation, Rotation(*Quaternion()),
	Scale, Scale(10.0, 10.0, 10.0, 10.0),
	MeshRef, MeshRef(0, pack_rgba8_srgb([0.3, 0.5, 0.7, 1.0])),
)

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
	pv = world.where(Position, Velocity)
	p, v = positions[pv], velocities[pv]
	p_vec, v_vec = p.vector(), v.vector()
	p_vec += v_vec * dt

	mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
	mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
	v_vec[mask_x, 0] *= -1
	v_vec[mask_z, 2] *= -1
	p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
	p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE

	p.set_vector(p_vec)
	v.set_vector(v_vec)


def render_system(world, instances):

	# NOTE: scale and tint are not modified often, so we could put them in a separated buffer
	# might want to keep data packed in cpu arrays too to avoid needing to pack when uploading the buffer
	# keeping as-is for now
	for mesh, offset, count, sl, entities, indirect_buf, cull_param_buf, cull_bg in draw_batches:
		instances[sl]["iPosition"] = positions[entities].vector()
		#instances[sl]["iTint"][:, 0] = mesh_refs[entities].tint
		#instances[sl]["iRotation"] = pack_quaternion(rotations[entities].vector())
		#instances[sl]["iScale"] = pack_scale(scales[entities].vector())
	
	instance_buffer.content = instances
	instance_buffer.resize(instances.size)

	uniform_buffer.content["view"] = camera.view()
	uniform_buffer.content["proj"] = camera.projection(RenderContext.aspect)
	uniform_buffer.content["light_dir"] = [*light_camera.direction(), 0.0]
	uniform_buffer.content["light_view_proj"] = light_camera.view() @ light_camera.projection(RenderContext.aspect) 

	instance_buffer.upload()
	uniform_buffer.upload()

	vp = camera.view() @ camera.projection(RenderContext.aspect)
	frustum_buffer.content["planes"] = extract_frustum_planes(vp)
	frustum_buffer.upload()
	clear_cull_cmd = RenderContext.commands("clear_cull")
	with (cull_cmd := RenderContext.commands("cull")).compute_pass(label="cull") as cp:
		cp.set_pipeline(cull_pipeline)
		for mesh, offset, count, sl, entities, indirect_buf, cull_param_buf, cull_bg in draw_batches:
			index_start, _ = mesh.index_range
			vertex_start, _ = mesh.vertex_range
			clear_cull_cmd.clear_buffer(indirect_buf, offset=4, size=4)
			#indirect_buf.write(np.array(
			#	[mesh.index_count, 0, index_start, vertex_start, offset],
			#	dtype=np.uint32,
			#))
			cp.set_bind_group(0, cull_bg)
			cp.dispatch((count + 63) // 64)

	with (shadow_cmd := RenderContext.commands("shadow")).render_pass(
		depth=shadow_texture.depth_attachment(clear=1.0),
		label="shadow",
	) as rp:
		rp.set_pipeline(shadow_pipeline)
		rp.set_bind_group(0, uniform_bindings)
		rp.set_bind_group(1, instance_bindings)
		for mesh, offset, count, sl, entities, indirect_buf, cull_param_buf, cull_bg in draw_batches:
			rp.set_vertex_buffer(0, mesh.vertex_buffer)
			rp.set_index_buffer(mesh.index_buffer, format=mesh.index_format)
			rp.draw_indexed_indirect(indirect_buf)

	with (main_cmd := RenderContext.commands("main")).render_pass(
		color=RenderContext.screen(clear=(0.02, 0.02, 0.03, 1.0)),
		depth=RenderContext.depth_attachment(clear=1.0),
		label="main",
	) as rp:
		rp.set_pipeline(main_pipeline)
		rp.set_bind_group(0, uniform_bindings)
		rp.set_bind_group(1, instance_bindings)
		rp.set_bind_group(2, shadow_bindings)
		for mesh, offset, count, sl, entities, indirect_buf, cull_param_buf, cull_bg in draw_batches:
			rp.set_vertex_buffer(0, mesh.vertex_buffer)
			rp.set_index_buffer(mesh.index_buffer, format=mesh.index_format)
			rp.draw_indexed_indirect(indirect_buf)

	RenderContext.submit(
		clear_cull_cmd.finish(),
		cull_cmd.finish(),
		shadow_cmd.finish(),
		main_cmd.finish(),
	)


all_renderables = world.where(Position, Rotation, Scale, MeshRef)
instances = np.empty(all_renderables.size,
	dtype=mesh_instance_dtype
)

instance_buffer = GpuBuffer(instances, BufferUsage.VERTEX | BufferUsage.STORAGE | BufferUsage.COPY_DST, label='instances')
visible_instances_buffer = GpuBuffer(np.zeros(all_renderables.size, dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST, label='visible_instances')

uniform_buffer = shader.UniformBuffer()

shadow_texture = create_depth_framebuffer(1024, 1024)
shadow_view = shadow_texture.view()
shadow_sampler = create_depth_sampler()

uniform_bindings = shader.bind_group(0, uniforms=uniform_buffer)
instance_bindings = shader.bind_group(1,
	instances=instance_buffer,
	visible_instances=visible_instances_buffer
)
shadow_bindings = shader.bind_group(
	2,
	shadow_map=shadow_view,
	shadow_sampler=shadow_sampler,
)


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
	instances[sl]["iPosition"] = positions[entities].vector()
	instances[sl]["iTint"][:, 0] = mesh_refs[entities].tint
	instances[sl]["iRotation"] = pack_quaternion(rotations[entities].vector())
	instances[sl]["iScale"] = pack_scale(scales[entities].vector())

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