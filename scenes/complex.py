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

cube_entities = world.create(200)
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

RenderContext.init_window(WINDOW_W, WINDOW_H, TITLE, target_fps=-1)
# RenderContext.capture_mouse()

shader = Shader(filepath='scenes/shaders/simple.shader', label="simple")
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

cube_mesh = RenderContext.resources["cube"]

meshes = {0: model_mesh, 1: cube_mesh}
draw_batches = []


def camera_system(camera, elapsed, camera_dist):
	cam_ang = elapsed * 0.5
	camera.position = Vec3((cos(cam_ang) * camera_dist, camera.position.y, sin(cam_ang) * camera_dist))


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
	for mesh, offset, count, sl, entities in draw_batches:
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

	with (shadow_cmd := RenderContext.commands("shadow")).render_pass(
		depth=shadow_texture.depth_attachment(clear=1.0),
		label="shadow",
	) as rp:
		rp.set_pipeline(shadow_pipeline)
		rp.set_bind_group(0, uniform_bindings)
		for mesh, offset, count, sl, entities in draw_batches:
			rp.draw_mesh(mesh, instances=instance_buffer, instance_offset=offset, instance_count=count)

	with (main_cmd := RenderContext.commands("main")).render_pass(
		color=RenderContext.screen(clear=(0.02, 0.02, 0.03, 1.0)),
		depth=RenderContext.depth_attachment(clear=1.0),
		label="main",
	) as rp:
		rp.set_pipeline(main_pipeline)
		rp.set_bind_group(0, uniform_bindings)
		rp.set_bind_group(1, shadow_bindings)
		for mesh, offset, count, sl, entities in draw_batches:
			rp.draw_mesh(mesh, instances=instance_buffer, instance_offset=offset, instance_count=count)

	RenderContext.submit(
		shadow_cmd.finish(),
		main_cmd.finish(),
	)


all_renderables = world.where(Position, Rotation, Scale, MeshRef)
instances = np.empty(all_renderables.size,
	dtype=mesh_instance_dtype
)
"""
mesh_instance_dtype = np.dtype([
    ("iPosition",  np.float32, 3), # 12 bytes
    ("iTint",      np.uint32,  1), # 4 bytes  (fills align 16 gap)
    ("iRotation",  np.uint32,  2), # 8 bytes  (snorm16 mapped to [-1, 1])
    ("iScale",     np.uint32,  2), # 8 bytes  (4x float16: x, y, z, 0)
])

NOTE: we might want to put mesh id in the last unused 16 bits of iScale
this would help for gpu frustum culling and indirect draw calls
"""

instance_buffer = GpuBuffer(instances, BufferUsage.VERTEX | BufferUsage.COPY_DST)

uniform_buffer = shader.UniformBuffer()

shadow_texture = create_depth_framebuffer(1024, 1024)
shadow_view = shadow_texture.view()
shadow_sampler = create_depth_sampler()

uniform_bindings = shader.bind_group(0, uniforms=uniform_buffer)
shadow_bindings = shader.bind_group(
	1,
	shadow_map=shadow_view,
	shadow_sampler=shadow_sampler,
)

offset = 0
for mesh_id, mesh in meshes.items():
	entities = all_renderables[mesh_refs[all_renderables].id == mesh_id]
	count = entities.size
	if count == 0: continue
	sl = slice(offset, offset + count)
	draw_batches.append((mesh, offset, count, sl, entities))
	offset += count
	# initialisation
	instances[sl]["iPosition"] = positions[entities].vector()
	instances[sl]["iTint"][:, 0] = mesh_refs[entities].tint
	instances[sl]["iRotation"] = pack_quaternion(rotations[entities].vector())
	instances[sl]["iScale"] = pack_scale(scales[entities].vector())
count

def clamp(val, val_min, val_max):
	return min(max(val, val_min), val_max)


def scroll_callback(xoff, yoff):
	global camera_dist, camera
	camera_dist = clamp(camera_dist - 5.0 * yoff, 5.0, 100.0)
	cp = camera.position
	y_factor = 1.0 if cp.y < 15.0 else 3.0
	camera.position = Vec3((cp.x, cp.y - y_factor * yoff, cp.z))
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