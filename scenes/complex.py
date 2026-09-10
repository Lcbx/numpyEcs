from RenderContext import *
from Utils import *
from ECS import *

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
	], dtype=np.uint32)
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
cull_frustum_pipeline = ComputePipeline(cull_shader, entry="cull_frustum", label="cull_frustum")
cull_hiz_pipeline = ComputePipeline(cull_shader, entry="cull_hiz", label="cull_hiz")

hzb_shader = Shader(filepath='scenes/shaders/hzb.shader', label="hzb")
hzb_depth_pipeline = ComputePipeline(hzb_shader, entry="reduce_depth", label="hzb_depth")
hzb_pipeline = ComputePipeline(hzb_shader, entry="main", label="hzb")

# --- HZB Texture Setup ---
draw_batches = []
hzb_size = None
hzb_passes = []

def resize_hzb(width, height):
	global hzb_size, depth_texture, hzb_texture, hzb_mip_views, num_mips, hzb_passes
	if hzb_size == (width, height): return
	hzb_size = (width, height)
	print(hzb_size)
	depth_texture = Texture(hzb_size, format="depth32float", usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING, label="prepass_depth")
	width, height = max(1, width // 2), max(1, height // 2)
	num_mips = max(width, height).bit_length()
	hzb_texture = Texture((width, height), format="r32float", usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING, mip_level_count=num_mips, label="hzb_pyramid")
	hzb_mip_views = [hzb_texture.view(base_mip_level=i, mip_level_count=1) for i in range(num_mips)]
	hzb_passes = [(hzb_depth_pipeline, hzb_shader.bind_group(0, src_prepass=depth_texture.view(), dst_depth=hzb_mip_views[0]), width, height)]
	for mip in range(1, num_mips):
		width, height = max(1, width // 2), max(1, height // 2)
		bindings = hzb_shader.bind_group(0, src_depth=hzb_mip_views[mip - 1], dst_depth=hzb_mip_views[mip])
		hzb_passes.append((hzb_pipeline, bindings, width, height))
	for *_, cull_hiz_bg in draw_batches:
		cull_hiz_bg.resources["hzb_texture"] = hzb_texture.view()

resize_hzb(WINDOW_W, WINDOW_H)

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
	Transform(Vec3(15.0, 0.0, 15.0), Vec3(10.0, 10.0, 10.0), Quaternion()),
	MeshRef(0, pack_rgba8_srgb([0.3, 0.5, 0.7, 1.0])),
)

cube_mesh = make_cube_mesh()

meshes = {0: model_mesh, 1: cube_mesh}

mesh_metadata_dtype = np.dtype([("box_center", np.float32, 4), ("box_extents", np.float32, 4)])
mesh_metadata = np.zeros(len(meshes), dtype=mesh_metadata_dtype)
mesh_metadata[0]["box_center"][:3], mesh_metadata[0]["box_extents"][:3] = compute_bounding_box(vertices)
mesh_metadata[1]["box_center"][:3], mesh_metadata[1]["box_extents"][:3] = compute_bounding_box(cube_mesh.vertices)
mesh_metadata_buffer = GpuBuffer(mesh_metadata, BufferUsage.STORAGE | BufferUsage.COPY_DST)

frustum_buffer = cull_shader.UniformBuffer("frustum")
frustum_bindings = cull_shader.bind_group(1, frustum=frustum_buffer)


def camera_system(camera, elapsed, camera_dist):
	cam_ang = elapsed * 0.5
	main_camera.position = Vec3(cos(cam_ang) * camera_dist, main_camera.position.y, sin(cam_ang) * camera_dist)


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
	resize_hzb(*RenderContext.windowDimensions)
	for mesh, offset, count, sl, entities, prepass_indirect_buf, main_indirect_buf, cull_param_buf, cull_frustum_bg, cull_hiz_bg in draw_batches:
		instances[sl]["iPosition"] = transforms[entities].position

	instance_buffer.content = instances
	instance_buffer.resize(instances.size)
	instance_buffer.upload()

	uniform_buffer.content["view"] = active_camera.view()
	uniform_buffer.content["proj"] = active_camera.projection(RenderContext.aspect)
	uniform_buffer.content["light_dir"] = [*light_camera.direction(), 0.0]
	uniform_buffer.content["light_view_proj"] = light_camera.view() @ light_camera.projection(RenderContext.aspect)
	uniform_buffer.upload()

	prepass_uniform_buffer.content["view"] = main_camera.view()
	prepass_uniform_buffer.content["proj"] = main_camera.projection(RenderContext.aspect)
	prepass_uniform_buffer.upload()

	vp = main_camera.view() @ main_camera.projection(RenderContext.aspect)
	frustum_buffer.content["planes"] = extract_frustum_planes(vp)
	frustum_buffer.upload()
	camera_params_buffer.content["view_proj"] = vp
	camera_params_buffer.upload()

	cmd = RenderContext.commands("frame_commands")

	# Reset instance counters for indirect draw targets
	for _, _, _, _, _, prepass_buf, main_buf, _, _, _ in draw_batches:
		cmd.clear_buffer(prepass_buf, offset=4, size=4)
		cmd.clear_buffer(main_buf, offset=4, size=4)

	# --- Step 1: Frustum Culling Pass ---
	with cmd.compute_pass(label="cull_frustum") as cp:
		cp.set_pipeline(cull_frustum_pipeline)
		cp.set_bind_group(1, frustum_bindings)
		for mesh, offset, count, sl, entities, prepass_buf, main_buf, cull_param_buf, cull_frustum_bg, cull_hiz_bg in draw_batches:
			cp.set_bind_group(0, cull_frustum_bg)
			cp.dispatch((count + 63) // 64)

	# --- Step 2: Depth Prepass ---
	depth_attachment = depth_texture.depth_attachment(clear=1.0)
	with cmd.render_pass(
		color=(),
		depth=depth_attachment,
		label="depth_prepass",
	) as rp:
		rp.set_pipeline(prepass_pipeline)
		rp.set_bind_group(0, prepass_uniform_bindings)
		rp.set_bind_group(1, prepass_instance_bindings)
		for mesh, offset, count, sl, entities, prepass_buf, main_buf, cull_param_buf, cull_frustum_bg, cull_hiz_bg in draw_batches:
			rp.set_vertex_buffer(0, mesh.vertex_buffer)
			rp.set_index_buffer(mesh.index_buffer, format=mesh.index_format)
			rp.draw_indexed_indirect(prepass_buf)

	# --- Step 3: Compute Hi-Z Downsampling ---
	for mip, (pipeline, bindings, width, height) in enumerate(hzb_passes):
		with cmd.compute_pass(label=f"hzb_mip_{mip}") as cp:
			cp.set_pipeline(pipeline)
			cp.set_bind_group(0, bindings)
			cp.dispatch((width + 15) // 16, (height + 15) // 16)

	# --- Step 4: Hi-Z Occlusion Culling Pass ---
	with cmd.compute_pass(label="cull_hiz") as cp:
		cp.set_pipeline(cull_hiz_pipeline)
		for mesh, offset, count, sl, entities, prepass_buf, main_buf, cull_param_buf, cull_frustum_bg, cull_hiz_bg in draw_batches:
			cp.set_bind_group(0, cull_frustum_bg)
			cp.set_bind_group(1, cull_hiz_bg)
			cp.dispatch((count + 63) // 64)

	# --- Step 5: Main Render Pass ---
	with cmd.render_pass(
		color=RenderContext.screen(clear=(0.02, 0.02, 0.03, 1.0)),
		depth=depth_texture.depth_attachment(clear=None if active_camera is main_camera else 1.0),
		label="main_pass",
	) as rp:
		rp.set_pipeline(main_pipeline)
		rp.set_bind_group(0, uniform_bindings)
		rp.set_bind_group(1, instance_bindings)
		for mesh, offset, count, sl, entities, prepass_buf, main_buf, cull_param_buf, cull_frustum_bg, cull_hiz_bg in draw_batches:
			rp.set_vertex_buffer(0, mesh.vertex_buffer)
			rp.set_index_buffer(mesh.index_buffer, format=mesh.index_format)
			rp.draw_indexed_indirect(main_buf)

	RenderContext.submit(cmd.finish())


all_renderables = world.where(Transform, MeshRef)
instances = np.empty(all_renderables.size, dtype=mesh_instance_dtype)

instance_buffer = GpuBuffer(
	instances,
	wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
	label='instances',
)
frustum_visible_instances_buffer = GpuBuffer(
	np.zeros(all_renderables.size, dtype=np.uint32),
	wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
	label='frustum_visible_instances',
)
main_visible_instances_buffer = GpuBuffer(
	np.zeros(all_renderables.size, dtype=np.uint32),
	wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
	label='main_visible_instances',
)

camera_params_buffer = cull_shader.UniformBuffer("camera_params")

uniform_buffer = shader.UniformBuffer()
uniform_bindings = shader.bind_group(0, uniforms=uniform_buffer)
prepass_uniform_buffer = shader.UniformBuffer()
prepass_uniform_bindings = shader.bind_group(0, uniforms=prepass_uniform_buffer)

prepass_instance_bindings = shader.bind_group(1, instances=instance_buffer, visible_instances=frustum_visible_instances_buffer)
instance_bindings = shader.bind_group(
	1,
	instances=instance_buffer,
	visible_instances=main_visible_instances_buffer,
)

offset = 0
for mesh_id, mesh in meshes.items():
	entities = all_renderables[mesh_refs[all_renderables].id == mesh_id]
	count = entities.size
	if count == 0:
		continue
	sl = slice(offset, offset + count)

	index_start, _ = mesh.index_range
	vertex_start, _ = mesh.vertex_range

	prepass_indirect_data = np.array(
		[mesh.index_count, 0, index_start, vertex_start, offset],
		dtype=np.uint32,
	)
	prepass_indirect_buf = GpuBuffer(
		prepass_indirect_data,
		wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST,
	)

	main_indirect_data = np.array(
		[mesh.index_count, 0, index_start, vertex_start, offset],
		dtype=np.uint32,
	)
	main_indirect_buf = GpuBuffer(
		main_indirect_data,
		wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST,
	)

	cull_param_data = np.array([mesh_id, offset, count, 0], dtype=np.uint32)
	cull_param_buf = GpuBuffer(cull_param_data, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

	cull_frustum_bg = cull_shader.bind_group(
		0,
		instances=instance_buffer,
		prepass_draw_cmd=prepass_indirect_buf,
		frustum_visible_instances=frustum_visible_instances_buffer,
		mesh_metadata=mesh_metadata_buffer,
		cull_params=cull_param_buf,
	)

	cull_hiz_bg = cull_shader.bind_group(
		1,
		camera_params=camera_params_buffer,
		hzb_texture=hzb_texture.view(),
		main_draw_cmd=main_indirect_buf,
		main_visible_instances=main_visible_instances_buffer,
	)

	draw_batches.append((
		mesh, offset, count, sl, entities,
		prepass_indirect_buf, main_indirect_buf, cull_param_buf,
		cull_frustum_bg, cull_hiz_bg
	))
	offset += count

	instances[sl]["iPosition"] = transforms[entities].position
	instances[sl]["iTint"][:, 0] = mesh_refs[entities].tint
	instances[sl]["iRotation"] = pack_quaternion(transforms[entities].rotation)
	instances[sl]["iScale"] = pack_scale(transforms[entities].scale)

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
	render_system(world, instances)