import ctypes
import os
import random as rd
import sys
import time
from math import cos, sin

import bgfx
import glfw
import numpy as np
from pygltflib import GLTF2, BufferView, Accessor
from pyrr import Matrix44 as Mat4, Vector3 as Vec3, Vector4 as Vec4, Quaternion

from ECS import *


BGFX_SHARED_LIB = os.environ.get("BGFX_SHARED_LIB") or "C:/Users/User/Desktop/tests/bgfx/bgfx/.build/win64_vs2022/bin/bgfx-shared-libRelease.dll"
if BGFX_SHARED_LIB:
	bgfx.load(BGFX_SHARED_LIB)


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
class Tint:
	value: np.uint32


def _get_data_from_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
	acc: Accessor = gltf.accessors[accessor_index]
	bv: BufferView = gltf.bufferViews[acc.bufferView]
	buf = gltf.buffers[bv.buffer]

	if buf.uri is None:
		bin_chunk = gltf.binary_blob()
	else:
		raise NotImplementedError("External buffers not handled")

	b = bv.byteOffset or 0
	e = b + (bv.byteLength or 0)
	view_bytes = memoryview(bin_chunk)[b:e]

	type_num_comps = {
		"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
		"MAT2": 4, "MAT3": 9, "MAT4": 16,
	}[acc.type]

	np_dtype = {
		5120: np.int8,
		5121: np.uint8,
		5122: np.int16,
		5123: np.uint16,
		5125: np.uint32,
		5126: np.float32,
	}[acc.componentType]

	stride = bv.byteStride or (np.dtype(np_dtype).itemsize * type_num_comps)
	count = acc.count
	offset = acc.byteOffset or 0
	tight = np.dtype(np_dtype).itemsize * type_num_comps

	if stride == tight:
		arr = np.frombuffer(view_bytes, dtype=np_dtype, count=count * type_num_comps, offset=offset)
		return arr.reshape(count, type_num_comps)

	rec = np.empty((count, type_num_comps), dtype=np_dtype)
	for i in range(count):
		start = offset + i * stride
		rec[i] = np.frombuffer(view_bytes, dtype=np_dtype, count=type_num_comps, offset=start)
	return rec


def load_gltf_first_mesh_interleaved(glb_path: str):
	gltf = GLTF2().load(glb_path)
	prim = gltf.meshes[0].primitives[0]

	pos = _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
	pos_count = pos.shape[0]
	nor = (
		_get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float32)
		if prim.attributes.NORMAL is not None else np.zeros_like(pos)
	)
	uv = (
		_get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float16)
		if prim.attributes.TEXCOORD_0 is not None else np.zeros((pos_count, 2), dtype=np.float16)
	)
	idx = _get_data_from_accessor(gltf, prim.indices).reshape(-1)
	idx = idx.astype(np.uint32 if pos_count.bit_length() > 16 else np.uint16)

	vertex_dtype = np.dtype([
		("position", np.float32, (3,)),
		("normal", np.float32, (3,)),
		("uv", np.float16, (2,)),
	], align=False)
	verts = np.empty(pos_count, dtype=vertex_dtype)
	verts["position"] = pos
	verts["normal"] = nor
	verts["uv"] = uv
	return verts, idx


CUBE_POSITIONS_24 = np.array((
	(-0.5,-0.5,+0.5), (+0.5,-0.5,+0.5), (+0.5,+0.5,+0.5), (-0.5,+0.5,+0.5),
	(+0.5,-0.5,-0.5), (-0.5,-0.5,-0.5), (-0.5,+0.5,-0.5), (+0.5,+0.5,-0.5),
	(+0.5,-0.5,+0.5), (+0.5,-0.5,-0.5), (+0.5,+0.5,-0.5), (+0.5,+0.5,+0.5),
	(-0.5,-0.5,-0.5), (-0.5,-0.5,+0.5), (-0.5,+0.5,+0.5), (-0.5,+0.5,-0.5),
	(-0.5,+0.5,+0.5), (+0.5,+0.5,+0.5), (+0.5,+0.5,-0.5), (-0.5,+0.5,-0.5),
	(-0.5,-0.5,-0.5), (+0.5,-0.5,-0.5), (+0.5,-0.5,+0.5), (-0.5,-0.5,+0.5),
), dtype=np.float32)

CUBE_NORMALS_24 = np.array(
	([(0.0, 0.0, 1.0)] * 4) +
	([(0.0, 0.0,-1.0)] * 4) +
	([(1.0, 0.0, 0.0)] * 4) +
	([(-1.0,0.0, 0.0)] * 4) +
	([(0.0, 1.0, 0.0)] * 4) +
	([(0.0,-1.0, 0.0)] * 4),
	dtype=np.float32,
)
CUBE_UVS_24 = np.array([(0.0,0.0), (1.0,0.0), (1.0,1.0), (0.0,1.0)] * 6, dtype=np.float16)
CUBE_INDICES_36 = np.array([
	0,1,2, 2,3,0,
	4,5,6, 6,7,4,
	8,9,10, 10,11,8,
	12,13,14, 14,15,12,
	16,17,18, 18,19,16,
	20,21,22, 22,23,20,
], dtype=np.uint16)


def make_cube_mesh_data():
	vertex_dtype = np.dtype([
		("position", np.float32, (3,)),
		("normal", np.float32, (3,)),
		("uv", np.float16, (2,)),
	], align=False)
	verts = np.empty(24, dtype=vertex_dtype)
	verts["position"] = CUBE_POSITIONS_24
	verts["normal"] = CUBE_NORMALS_24
	verts["uv"] = CUBE_UVS_24
	return verts, CUBE_INDICES_36


def linear_to_srgb(x):
	x = np.asarray(x, dtype=np.float32)
	return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


_RGBA_SHIFT = np.array([0, 8, 16, 24], dtype=np.uint32)
def pack_rgba8_srgb(rgba):
	rgba = np.array([*linear_to_srgb(rgba[:3]), rgba[3]])
	rgba8 = np.rint(rgba * 255).astype(np.uint32)
	return np.uint32(np.sum(rgba8 << _RGBA_SHIFT))


def unpack_rgba8(packed):
	packed = np.asarray(packed, dtype=np.uint32).reshape(-1)
	out = np.empty((packed.size, 4), dtype=np.float32)
	out[:, 0] = ((packed >> 0) & 0xff) / 255.0
	out[:, 1] = ((packed >> 8) & 0xff) / 255.0
	out[:, 2] = ((packed >> 16) & 0xff) / 255.0
	out[:, 3] = ((packed >> 24) & 0xff) / 255.0
	return out


class Camera:
	def __init__(self, position, target, up, fovy_deg, near=0.1, far=1000.0, perspective=True):
		self.position = Vec3(position)
		self.target = Vec3(target)
		self.up = Vec3(up)
		self.fovy_deg = fovy_deg
		self.near = near
		self.far = far
		self.perspective = perspective

	def view(self):
		return np.asarray(Mat4.look_at(self.position, self.target, self.up), dtype=np.float32)

	def projection(self, aspect, homogeneous_depth):
		if self.perspective:
			return perspective_projection(self.fovy_deg, aspect, self.near, self.far, homogeneous_depth)
		top = self.fovy_deg * 0.5
		right = top * aspect
		return orthogonal_projection(-right, right, -top, top, self.near, self.far, homogeneous_depth)

	def direction(self):
		l = self.position - self.target
		return l / np.linalg.norm(l)


def perspective_projection(fovy_deg, aspect, near, far, homogeneous_depth):
	f = 1.0 / np.tan(np.deg2rad(fovy_deg) * 0.5)
	if homogeneous_depth:
		zz = (far + near) / (near - far)
		zw = 2.0 * far * near / (near - far)
	else:
		zz = far / (near - far)
		zw = far * near / (near - far)
	return np.array([
		[f / aspect, 0.0, 0.0, 0.0],
		[0.0, f, 0.0, 0.0],
		[0.0, 0.0, zz, -1.0],
		[0.0, 0.0, zw, 0.0],
	], dtype=np.float32)


def orthogonal_projection(left, right, bottom, top, near, far, homogeneous_depth):
	rml = right - left
	rpl = right + left
	tmb = top - bottom
	tpb = top + bottom
	fn = far - near
	if homogeneous_depth:
		zz = -2.0 / fn
		zw = -(far + near) / fn
	else:
		zz = -1.0 / fn
		zw = -near / fn
	return np.array([
		[2.0/rml, 0.0, 0.0, 0.0],
		[0.0, 2.0/tmb, 0.0, 0.0],
		[0.0, 0.0, zz, 0.0],
		[-rpl/rml, -tpb/tmb, zw, 1.0],
	], dtype=np.float32)


def ptr(array):
	array = np.asarray(array)
	if not array.flags.c_contiguous:
		raise ValueError("expected contiguous array")
	return ctypes.c_void_p(array.ctypes.data)


def bgfx_mem(array):
	array = np.ascontiguousarray(array)
	return bgfx.bgfx_copy(ctypes.c_void_p(array.ctypes.data), array.nbytes)


def make_vertex_layout():
	layout = bgfx.VertexLayout()
	bgfx.bgfx_vertex_layout_begin(ctypes.byref(layout), bgfx.bgfx_get_renderer_type())
	bgfx.bgfx_vertex_layout_add(ctypes.byref(layout), bgfx.Attrib.Position, 3, bgfx.AttribType.Float, False, False)
	bgfx.bgfx_vertex_layout_add(ctypes.byref(layout), bgfx.Attrib.Normal, 3, bgfx.AttribType.Float, False, False)
	bgfx.bgfx_vertex_layout_add(ctypes.byref(layout), bgfx.Attrib.TexCoord0, 2, bgfx.AttribType.Half, False, False)
	bgfx.bgfx_vertex_layout_end(ctypes.byref(layout))
	return layout


def create_mesh(vertices, indices, layout):
	vertices = np.ascontiguousarray(vertices)
	indices = np.ascontiguousarray(indices)
	index_flags = int(bgfx.BufferFlags.Index32) if indices.dtype == np.uint32 else 0
	return {
		"vb": bgfx.bgfx_create_vertex_buffer(bgfx_mem(vertices), ctypes.byref(layout), 0),
		"ib": bgfx.bgfx_create_index_buffer(bgfx_mem(indices), index_flags),
		"vertices": len(vertices),
		"indices": indices.size,
	}


def renderer_shader_dir(renderer_type):
	return {
		bgfx.RendererType.Direct3D11: "dx11",
		bgfx.RendererType.Direct3D12: "dx11",
		bgfx.RendererType.OpenGL: "glsl",
		bgfx.RendererType.OpenGLES: "essl",
		bgfx.RendererType.Vulkan: "spirv",
		bgfx.RendererType.Metal: "metal",
		bgfx.RendererType.WebGPU: "wgsl",
	}.get(bgfx.RendererType(renderer_type))


def load_shader(path):
	data = np.fromfile(path, dtype=np.uint8)
	return bgfx.bgfx_create_shader(bgfx_mem(data))


def load_program(shader_dir, vertex_name, fragment_name):
	vsh = load_shader(os.path.join(shader_dir, vertex_name + ".bin"))
	fsh = load_shader(os.path.join(shader_dir, fragment_name + ".bin"))
	return bgfx.bgfx_create_program(vsh, fsh, True)


def set_platform_data(init, window):
	if sys.platform == "win32":
		init.platformData.nwh = glfw.get_win32_window(window)
		return
	if sys.platform == "darwin":
		init.platformData.nwh = glfw.get_cocoa_window(window)
		return
	if sys.platform.startswith("linux"):
		platform = glfw.get_platform() if hasattr(glfw, "get_platform") else glfw.PLATFORM_X11
		if platform == glfw.PLATFORM_WAYLAND:
			init.platformData.ndt = glfw.get_wayland_display()
			init.platformData.nwh = glfw.get_wayland_window(window)
			init.platformData.type = int(bgfx.NativeWindowHandleType.Wayland)
		else:
			init.platformData.ndt = glfw.get_x11_display()
			init.platformData.nwh = glfw.get_x11_window(window)
		return
	raise RuntimeError(f"Unsupported platform: {sys.platform}")


def make_world_instances(entities, positions, rotations, scales, tints):
	position = positions[entities].vector().astype(np.float32, copy=False)
	rotation = rotations[entities].vector().astype(np.float32, copy=False)
	scale = scales[entities].vector().astype(np.float32, copy=False)
	tint = tints[entities].value

	out = np.zeros((entities.size, 16), dtype=np.float32)
	out[:, 0:3] = position[:, 0:3]
	out[:, 4:8] = rotation[:, 0:4]
	out[:, 8:11] = scale[:, 0:3]
	out[:, 12:16] = unpack_rgba8(tint)
	return out


def set_instance(out, index, position, rotation, scale, tint):
	out[index, :] = 0.0
	out[index, 0:3] = position
	out[index, 4:8] = rotation
	out[index, 8:11] = scale[:3]
	out[index, 12:16] = unpack_rgba8([tint])[0]


def alloc_instance_buffer(instances):
	instances = np.ascontiguousarray(instances, dtype=np.float32)
	count = len(instances)
	stride = instances.shape[1] * 4
	available = bgfx.bgfx_get_avail_instance_data_buffer(count, stride)
	if available < count:
		raise RuntimeError(f"bgfx transient instance buffer exhausted: {available}/{count}")
	idb = bgfx.InstanceDataBuffer()
	bgfx.bgfx_alloc_instance_data_buffer(ctypes.byref(idb), count, stride)
	ctypes.memmove(idb.data, instances.ctypes.data, instances.nbytes)
	return idb


def submit_mesh(view_id, mesh, program, idb, state, shadow_texture=None, shadow_sampler=None):
	bgfx.bgfx_set_vertex_buffer(0, mesh["vb"], 0, mesh["vertices"])
	bgfx.bgfx_set_index_buffer(mesh["ib"], 0, mesh["indices"])
	bgfx.bgfx_set_instance_data_buffer(ctypes.byref(idb), 0, idb.num)
	if shadow_texture is not None:
		bgfx.bgfx_set_texture(0, shadow_sampler, shadow_texture, 0xffffffff)
	bgfx.bgfx_set_state(state, 0)
	bgfx.bgfx_submit(view_id, program, 0, int(bgfx.DiscardFlags.All))


def clamp(val, val_min, val_max):
	return min(max(val, val_min), val_max)


world = ECS()
world.register(Position, Velocity, Rotation, Scale, Tint)

positions = world.get(Position)
velocities = world.get(Velocity)
rotations = world.get(Rotation)
scales = world.get(Scale)
tints = world.get(Tint)

SPACE_SIZE = 180
CUBE_MAX_SIDE = 7

ground = world.create()
world.add(
	ground,
	Position, Position(0, -0.51, 0),
	Rotation, Rotation(*Quaternion()),
	Scale, Scale(2 * SPACE_SIZE, 1, 2 * SPACE_SIZE, 0),
	Tint, Tint(pack_rgba8_srgb([0.5, 0.5, 0.5, 1.0])),
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
cube_tints = np.asarray([
	pack_rgba8_srgb([rd.random(), rd.random(), rd.random(), 1.0])
	for _ in range(cube_count)
], dtype=np.uint32)

world.add(
	cube_entities,
	Position, cube_positions,
	Velocity, cube_velocities,
	Rotation, cube_rotations,
	Scale, cube_scales,
	Tint, cube_tints,
)

WINDOW_W, WINDOW_H = 1200, 1200
TITLE = "glfw + bgfx"
SHADOW_SIZE = 1024
SHADOW_VIEW = 0
MAIN_VIEW = 1

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

if not glfw.init():
	raise RuntimeError("glfw.init() failed")

glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
window = glfw.create_window(WINDOW_W, WINDOW_H, TITLE, None, None)
if not window:
	glfw.terminate()
	raise RuntimeError("glfw.create_window() failed")

framebuffer_w, framebuffer_h = glfw.get_framebuffer_size(window)
reset_flags = int(bgfx.ResetFlags.None_)

init = bgfx.Init()
bgfx.bgfx_init_ctor(ctypes.byref(init))
init.type = int(bgfx.RendererType.Count)
init.resolution.width = framebuffer_w
init.resolution.height = framebuffer_h
init.resolution.reset = reset_flags
set_platform_data(init, window)

if not bgfx.bgfx_init(ctypes.byref(init)):
	glfw.destroy_window(window)
	glfw.terminate()
	raise RuntimeError("bgfx_init() failed")

caps = bgfx.bgfx_get_caps().contents
if not caps.supported & int(bgfx.CapsFlags.Instancing):
	raise RuntimeError("Selected bgfx renderer does not support instancing")
if not caps.supported & int(bgfx.CapsFlags.TextureCompareLequal):
	raise RuntimeError("Selected bgfx renderer does not support comparison shadow samplers")

renderer_dir = renderer_shader_dir(bgfx.bgfx_get_renderer_type())
if renderer_dir is None:
	raise RuntimeError(f"No shader directory mapping for renderer {bgfx.bgfx_get_renderer_type()}")
shader_dir = os.path.join(os.path.dirname(__file__), "shaders", renderer_dir)

main_program = load_program(shader_dir, "vs_main", "fs_main")
shadow_program = load_program(shader_dir, "vs_shadow", "fs_shadow")

u_light_mtx = bgfx.bgfx_create_uniform(b"u_lightMtx", bgfx.UniformType.Mat4, 1)
u_light_dir = bgfx.bgfx_create_uniform(b"u_lightDir", bgfx.UniformType.Vec4, 1)
s_shadow_map = bgfx.bgfx_create_uniform(b"s_shadowMap", bgfx.UniformType.Sampler, 1)

vertex_layout = make_vertex_layout()
vertices, indices = load_gltf_first_mesh_interleaved("scenes/resources/rooftop_utility_pole.glb")
model_mesh = create_mesh(vertices, indices, vertex_layout)
cube_mesh = create_mesh(*make_cube_mesh_data(), vertex_layout)

model_instances = np.zeros((2, 16), dtype=np.float32)
model_scale = 10.0
set_instance(
	model_instances, 0,
	[15.0, 0.0, 15.0], Quaternion(), [model_scale] * 4,
	pack_rgba8_srgb(Vec4([0.3, 0.5, 0.7, 1.0])),
)

render_entities = world.where(Position, Rotation, Scale, Tint)

shadow_flags = int(
	bgfx.TextureFlags.Rt |
	bgfx.SamplerFlags.CompareLequal |
	bgfx.SamplerFlags.UClamp |
	bgfx.SamplerFlags.VClamp
)
shadow_texture = bgfx.bgfx_create_texture_2d(
	SHADOW_SIZE, SHADOW_SIZE, False, 1,
	bgfx.TextureFormat.D16, shadow_flags, None, 0,
)
shadow_handles = (bgfx.TextureHandle * 1)(shadow_texture)
shadow_framebuffer = bgfx.bgfx_create_frame_buffer_from_handles(1, shadow_handles, True)

shadow_state = int(
	bgfx.StateFlags.WriteZ |
	bgfx.StateFlags.DepthTestLess |
	bgfx.StateFlags.CullCw
)
main_state = int(
	bgfx.StateFlags.WriteRgb |
	bgfx.StateFlags.WriteA |
	bgfx.StateFlags.WriteZ |
	bgfx.StateFlags.DepthTestLess |
	bgfx.StateFlags.CullCw
)

bgfx.bgfx_set_view_clear(SHADOW_VIEW, int(bgfx.ClearFlags.Depth), 0, 1.0, 0)
bgfx.bgfx_set_view_clear(MAIN_VIEW, int(bgfx.ClearFlags.Color | bgfx.ClearFlags.Depth), 0x050508ff, 1.0, 0)
bgfx.bgfx_set_view_frame_buffer(SHADOW_VIEW, shadow_framebuffer)


def scroll_callback(window, xoff, yoff):
	global camera_dist
	camera_dist = clamp(camera_dist - 5.0 * yoff, 5.0, 100.0)
	cp = camera.position
	y_factor = 1.0 if cp.y < 15.0 else 3.0
	camera.position = Vec3((cp.x, cp.y - y_factor * yoff, cp.z))


glfw.set_scroll_callback(window, scroll_callback)

start_t = time.perf_counter()
prev_t = start_t
fps_frames = 0
fps_print_timestamp = start_t

try:
	while not glfw.window_should_close(window):
		glfw.poll_events()
		now = time.perf_counter()
		frame_time = now - prev_t
		prev_t = now

		fps_frames += 1
		if now - fps_print_timestamp >= 1.0:
			print(f"fps {fps_frames}")
			fps_frames = 0
			fps_print_timestamp = now

			set_instance(
				model_instances, 1,
				[np.random.rand() * 15.0, 0.0, np.random.rand() * 15.0],
				Quaternion(), [model_scale] * 4,
				pack_rgba8_srgb(Vec4([0.6, 0.5, 0.4, 1.0])),
			)

		elapsed = now - start_t
		cam_ang = elapsed * 0.5
		camera.position = Vec3((
			cos(cam_ang) * camera_dist,
			camera.position.y,
			sin(cam_ang) * camera_dist,
		))

		pv = world.where(Position, Velocity)
		p = positions[pv]
		v = velocities[pv]
		p_vec, v_vec = p.vector(), v.vector()
		p_vec += v_vec * frame_time

		mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
		mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
		v_vec[mask_x, 0] *= -1
		v_vec[mask_z, 2] *= -1
		p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
		p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE
		p.set_vector(p_vec)
		v.set_vector(v_vec)

		world_instances = make_world_instances(render_entities, positions, rotations, scales, tints)
		world_idb = alloc_instance_buffer(world_instances)
		model_idb = alloc_instance_buffer(model_instances)

		new_w, new_h = glfw.get_framebuffer_size(window)
		if new_w != framebuffer_w or new_h != framebuffer_h:
			framebuffer_w, framebuffer_h = max(1, new_w), max(1, new_h)
			bgfx.bgfx_reset(framebuffer_w, framebuffer_h, reset_flags, bgfx.TextureFormat.Count)
			bgfx.bgfx_set_view_frame_buffer(SHADOW_VIEW, shadow_framebuffer)

		aspect = framebuffer_w / framebuffer_h
		view = np.ascontiguousarray(camera.view(), dtype=np.float32)
		proj = np.ascontiguousarray(camera.projection(aspect, caps.homogeneousDepth), dtype=np.float32)
		light_view = np.ascontiguousarray(light_camera.view(), dtype=np.float32)
		light_proj = np.ascontiguousarray(light_camera.projection(1.0, caps.homogeneousDepth), dtype=np.float32)

		sy = 0.5 if caps.originBottomLeft else -0.5
		sz = 0.5 if caps.homogeneousDepth else 1.0
		tz = 0.5 if caps.homogeneousDepth else 0.0
		crop = np.array([
			[0.5, 0.0, 0.0, 0.0],
			[0.0, sy, 0.0, 0.0],
			[0.0, 0.0, sz, 0.0],
			[0.5, 0.5, tz, 1.0],
		], dtype=np.float32)
		light_mtx = np.ascontiguousarray(light_view @ light_proj @ crop, dtype=np.float32)
		light_dir = np.ascontiguousarray([*light_camera.direction(), 0.0], dtype=np.float32)

		bgfx.bgfx_set_view_rect(SHADOW_VIEW, 0, 0, SHADOW_SIZE, SHADOW_SIZE)
		bgfx.bgfx_set_view_transform(SHADOW_VIEW, ptr(light_view), ptr(light_proj))
		bgfx.bgfx_set_view_rect(MAIN_VIEW, 0, 0, framebuffer_w, framebuffer_h)
		bgfx.bgfx_set_view_transform(MAIN_VIEW, ptr(view), ptr(proj))

		bgfx.bgfx_touch(SHADOW_VIEW)
		bgfx.bgfx_touch(MAIN_VIEW)

		submit_mesh(SHADOW_VIEW, model_mesh, shadow_program, model_idb, shadow_state)
		submit_mesh(SHADOW_VIEW, cube_mesh, shadow_program, world_idb, shadow_state)

		bgfx.bgfx_set_uniform(u_light_mtx, ptr(light_mtx), 1)
		bgfx.bgfx_set_uniform(u_light_dir, ptr(light_dir), 1)
		submit_mesh(MAIN_VIEW, model_mesh, main_program, model_idb, main_state, shadow_texture, s_shadow_map)

		bgfx.bgfx_set_uniform(u_light_mtx, ptr(light_mtx), 1)
		bgfx.bgfx_set_uniform(u_light_dir, ptr(light_dir), 1)
		submit_mesh(MAIN_VIEW, cube_mesh, main_program, world_idb, main_state, shadow_texture, s_shadow_map)

		bgfx.bgfx_frame(0)
finally:
	bgfx.bgfx_destroy_program(main_program)
	bgfx.bgfx_destroy_program(shadow_program)
	bgfx.bgfx_destroy_uniform(u_light_mtx)
	bgfx.bgfx_destroy_uniform(u_light_dir)
	bgfx.bgfx_destroy_uniform(s_shadow_map)
	bgfx.bgfx_destroy_vertex_buffer(model_mesh["vb"])
	bgfx.bgfx_destroy_index_buffer(model_mesh["ib"])
	bgfx.bgfx_destroy_vertex_buffer(cube_mesh["vb"])
	bgfx.bgfx_destroy_index_buffer(cube_mesh["ib"])
	bgfx.bgfx_destroy_frame_buffer(shadow_framebuffer)
	bgfx.bgfx_shutdown()
	glfw.destroy_window(window)
	glfw.terminate()
