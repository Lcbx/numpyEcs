from RenderContext import *
from Utils import *
from ECS import *
from DrawBatches import *

from math import cos, sin
import random as rd


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
	shader_id: np.uint32

world = ECS()
transforms, velocities, mesh_refs = world.register(
	Transform, Velocity, MeshRef
)

CUBE_COUNT = 10000
SPACE_SIZE = 180
CUBE_MAX_SIDE = 7

ground = world.create()
world.add(
	ground,
	Transform(
		Vec3(0, -0.51, 0),
		Vec3(2 * SPACE_SIZE, 1, 2 * SPACE_SIZE),
		Quaternion(),
	),
	MeshRef(1, pack_rgba8_srgb([0.5, 0.5, 0.5, 1.0]), 0),
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
	Quaternion.from_axis_rotation((0.0, 1.0, 0.0), rd.random() * 3.143)
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
	], dtype=np.uint32),
	np.zeros(cube_count, dtype=np.uint32),
))

world.add(
	cube_entities,
	Transform, (cube_positions, cube_scales, cube_rotations),
	Velocity, cube_velocities,
	MeshRef, cube_meshrefs,
)

WINDOW_W, WINDOW_H = 1200, 1200
TITLE = "glfw + wgpu - Split Compute Hi-Z Pipeline"

camera_dist = 30.0
main_camera = Camera(
	position=(-20.0, 70.0, 25.0),
	target=(0.0, 10.0, 0.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=60.0,
	near=0.1,
	far=1000.0,
)
observer_camera = Camera(
	position=(140.0, 170.0, 140.0),
	target=(0.0, 0.0, 0.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=60.0,
	near=0.1,
	far=1000.0,
)
active_camera = main_camera

light_camera = Camera(
	position=(-30.0, 30.0, 25.0),
	target=(0.0, 0.0, -20.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=90.0,
	near=1.0,
	far=300.0,
	perspective=False,
)

RenderContext.init_window(
	WINDOW_W, WINDOW_H, TITLE,
	target_fps=-1,
	required_gpu_features=["indirect-first-instance"]
)

# --- Shaders & Pipelines ---
shader = Shader(filepath='scenes/shaders/complex.shader', label="complex")
prepass_pipeline = RenderPipeline(
	shader,
	vertex_entry="vertex",
	fragment_entry=None,
	label="prepass",
)
main_pipeline = RenderPipeline(
	shader,
	vertex_entry="vertex",
	fragment_entry="fragment",
	depth_test="less-equal",
	label="main",
)

cull_shader = Shader(filepath='scenes/shaders/cull.shader', label="cull")
reset_draw_pipeline = ComputePipeline(cull_shader, entry="reset_draw_counts", label="reset_draw_counts")
cull_frustum_pipeline = ComputePipeline(cull_shader, entry="cull_frustum", label="cull_frustum")
cull_hiz_pipeline = ComputePipeline(cull_shader, entry="cull_hiz", label="cull_hiz")

hzb_shader = Shader(filepath='scenes/shaders/hzb.shader', label="hzb")
hzb_depth_pipeline = ComputePipeline(hzb_shader, entry="reduce_depth", label="hzb_depth")
hzb_pipeline = ComputePipeline(hzb_shader, entry="main", label="hzb")

# --- Draw registries & HZB ---
render_data = create_draw_data(cull_shader)
register_shader(render_data, 0, RenderShader(ShaderPass(prepass_pipeline), ShaderPass(main_pipeline)))
hzb = HZB()


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
	Transform(Vec3(15.0, 0.0, 15.0), Vec3(10.0, 10.0, 10.0), Quaternion()),
	MeshRef(0, pack_rgba8_srgb([0.3, 0.5, 0.7, 1.0]), 0),
)

cube_mesh = make_cube_mesh()

register_mesh(render_data, 0, model_mesh)
register_mesh(render_data, 1, cube_mesh)


def camera_system(camera, elapsed, camera_dist):
	cam_ang = elapsed * 0.5
	camera.position = Vec3(cos(cam_ang) * camera_dist, camera.position.y, sin(cam_ang) * camera_dist)


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


def update_instances(world, data):
	sync_batches(data, world, Transform, MeshRef)
	version = (transforms.version("position"), mesh_refs.version("tint"), transforms.version("rotation"), transforms.version("scale"))
	if data.instance_version == version: return
	previous = data.instance_version or (None,) * 4
	changed = tuple(old != new for old, new in zip(previous, version))
	count = data.entities.size
	if count:
		buffer = data.buffers["instances"]
		instances = buffer.content[:count]
		if changed[0] or changed[2] or changed[3]: p = transforms[data.entities]
		if changed[0]: instances["iPosition"] = p.position
		if changed[1]: instances["iTint"] = np.asarray(mesh_refs[data.entities].tint).reshape(instances["iTint"].shape)
		if changed[2]: instances["iRotation"] = pack_quaternion(p.rotation)
		if changed[3]: instances["iScale"] = pack_scale(p.scale)
		buffer.upload_range(0, count)
	data.instance_version = version


def update_cameras(data, culling_camera, rendering_camera, light_dir):
	view = culling_camera.view()
	proj = culling_camera.projection(RenderContext.aspect)
	for buffer, camera in ((data.prepass_uniform_buffer, culling_camera), (data.uniform_buffer, rendering_camera)):
		buffer.content["view"] = view if camera is culling_camera else camera.view()
		buffer.content["proj"] = proj if camera is culling_camera else camera.projection(RenderContext.aspect)
		buffer.content["light_dir"] = [*light_dir, 0.0]
		buffer.upload()
	vp = view @ proj
	buffer = data.camera_params_buffer
	buffer.content["view_proj"] = vp
	buffer.content["planes"] = extract_frustum_planes(vp)
	buffer.content["workgroup_count"] = data.workgroup_count
	buffer.content["batch_count"] = len(data.draw_batches)
	buffer.upload()


def render_system(world, culling_camera, rendering_camera):
	width, height = RenderContext.windowDimensions
	if width <= 0 or height <= 0: return
	update_instances(world, render_data)
	resize_hzb(hzb, (width, height), hzb_shader, (hzb_depth_pipeline, hzb_pipeline))
	refresh_bindings(render_data, cull_shader, hzb)
	update_cameras(render_data, culling_camera, rendering_camera, light_camera.direction())
	cmd = RenderContext.commands("frame_commands")
	reset_draw_counts(cmd, render_data, reset_draw_pipeline)

	# --- Step 1: Frustum Culling Pass ---
	cull_instances(cmd, render_data, cull_frustum_pipeline, "frustum")

	# --- Step 2: Depth Prepass ---
	with cmd.render_pass(color=(), depth=hzb.depth_texture.depth_attachment(clear=1.0), label="depth_prepass") as rp:
		draw_batches(rp, render_data, "prepass")

	# --- Step 3: Compute Hi-Z Downsampling ---
	build_hzb(cmd, hzb)

	# --- Step 4: Hi-Z Occlusion Culling Pass ---
	cull_instances(cmd, render_data, cull_hiz_pipeline, "hiz")

	# --- Step 5: Main Render Pass ---
	depth = hzb.depth_texture.depth_attachment(clear=None if rendering_camera is culling_camera else 1.0)
	with cmd.render_pass(color=RenderContext.screen(clear=(0.02, 0.02, 0.03, 1.0)), depth=depth, label="main_pass") as rp:
		draw_batches(rp, render_data, "main")
	RenderContext.submit(cmd.finish())


def clamp(val, val_min, val_max):
	return min(max(val, val_min), val_max)

def scroll_callback(xoff, yoff):
	global camera_dist, main_camera
	camera_dist = clamp(camera_dist - 5.0 * yoff, 5.0, 100.0)
	cp = main_camera.position
	y_factor = 1.0 if cp.y < 15.0 else 3.0
	main_camera.position = Vec3(cp.x, cp.y - y_factor * yoff, cp.z)
	return True

def on_char(code):
	global active_camera
	if chr(code) == " ":
		active_camera = observer_camera if active_camera is main_camera else main_camera
		print("switch")
	return True

RenderContext.event_handlers["mouse_scroll"].append(scroll_callback)
RenderContext.event_handlers["char"].append(on_char)

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
	camera_system(main_camera, elapsed, camera_dist)
	movement_system(world, RenderContext.frame_time)
	render_system(world, main_camera, active_camera)
