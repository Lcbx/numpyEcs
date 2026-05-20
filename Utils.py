from RenderContext import *

import numpy as np
from pyrr import Matrix44 as Mat4, Vector3 as Vec3, Vector4 as Vec4, Quaternion
from pygltflib import GLTF2, BufferView, Accessor

Vec3_f32_type = np.dtype( (np.float32, (3,))  )
Vec4_f32_type = np.dtype( (np.float32, (4,))  )
Vec4_f16_type = np.dtype( (np.float16, (4,))  )
Mat4_f32_type = np.dtype( (np.float32, (4,4)) )
Color_type    = np.dtype( (np.uint32,  (1,))  )

class Camera:
	def __init__(self, position: tuple, target: tuple, up: tuple, fovy_deg:float, near:float=0.1, far:float=1000.0, perspective:bool=True):
		self.position = Vec3(position)
		self.target = Vec3(target)
		self.up = Vec3(up)
		self.fovy_deg = fovy_deg
		self.near = near
		self.far = far
		self.perspective = perspective

	def view(self) -> Mat4:
		return Mat4.look_at(self.position, self.target, self.up)

	def projection(self, aspect) -> Mat4:
		if self.perspective:
			return Mat4.perspective_projection( self.fovy_deg, aspect, self.near, self.far )

		top = self.fovy_deg * 0.5;
		right = top*aspect;
		return Mat4.orthogonal_projection(-right, right, top, -top, self.near, self.far)


def _get_data_from_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
	acc: Accessor = gltf.accessors[accessor_index]
	bv: BufferView = gltf.bufferViews[acc.bufferView]
	buf = gltf.buffers[bv.buffer]

	if buf.uri is None:  # GLB
		bin_chunk = gltf.binary_blob()
	else:
		raise NotImplementedError("External buffers not handled")

	b = bv.byteOffset or 0
	e = b + (bv.byteLength or 0)
	view_bytes = memoryview(bin_chunk)[b:e]

	type_num_comps = {
		"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16
	}[acc.type]

	np_dtype = {
		5120: np.int8,
		5121: np.uint8,
		5122: np.int16,
		5123: np.uint16,
		5125: np.uint32,
		5126: np.float32
	}[acc.componentType]

	stride = bv.byteStride or (np.dtype(np_dtype).itemsize * type_num_comps)
	count = acc.count
	offset = acc.byteOffset or 0
	tight = np.dtype(np_dtype).itemsize * type_num_comps

	arr = np.frombuffer(view_bytes, dtype=np_dtype, count=count * type_num_comps, offset=offset)
	if stride != tight:
		rec = np.empty((count, type_num_comps), dtype=np_dtype)
		base = offset
		for i in range(count):
			start = base + i * stride
			rec[i] = np.frombuffer(view_bytes, dtype=np_dtype, count=type_num_comps, offset=start)
		return rec
	else:
		return arr.reshape(count, type_num_comps)



def load_gltf_first_mesh(glb_path: str) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
	gltf = GLTF2().load(glb_path)
	mesh = gltf.meshes[0]
	prim = mesh.primitives[0]
	
	#print(f'{mesh=}')

	idx =   _get_data_from_accessor(gltf, prim.indices)
	pos =   _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
	nor = ( _get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float32)
		if prim.attributes.NORMAL is not None
		else np.zeros_like( pos.shape, dtype=np.float32)
	)
	uv  = ( _get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float16)
		if prim.attributes.TEXCOORD_0 is not None
		else np.zeros( (pos.shape[0], 2), dtype=np.float16)
	)

	return pos, nor, uv, idx


def load_gltf_first_mesh_interleaved(glb_path: str) -> Tuple[np.ndarray,np.ndarray]:
	( pos, nor, uv, idx ) = load_gltf_first_mesh(glb_path)
	verts = interleave_mesh_position_normal_uv(pos, nor, uv)
	return verts, idx.astype(np.uint32).reshape(-1)

def interleave_mesh_position_normal_uv(pos, nor, uv):
	vertex_dtype = np.dtype([
			("position", np.float32, (3,)),
			("normal",   np.float32, (3,)), # float16x3 does not exist in wgpu
			("uv",	   	 np.float16, (2,)),
		], align=False,   # important: keep it tightly packed (no padding surprises)
	)
	#print("stride:", vertex_dtype.itemsize)
	#print("offsets:", {n: vertex_dtype.fields[n][1] for n in vertex_dtype.names})

	verts = np.empty(pos.shape[0], dtype=vertex_dtype)
	verts["position"] = pos
	verts["normal"]   = nor
	verts["uv"]		  = uv
	return verts


""" # not perfect
def load_gltf_meshes(program : Program, glb_path: str)-> IndexedVertexList:
	gltf = GLTF2().load(glb_path)

	meshes = []
	for mesh in gltf.meshes:
		for prim in mesh.primitives:
			pos = _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
			nor = _get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float32) if prim.attributes.NORMAL is not None else np.zeros_like(pos)
			uv  = _get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float32) if prim.attributes.TEXCOORD_0 is not None else np.zeros((pos.shape[0],2), dtype=np.float32)
			idx = _get_data_from_accessor(gltf, prim.indices)
			if idx.dtype != np.uint32:
				idx = idx.astype(np.uint32)

		model = Mesh(program, pos, nor, uv, idx)
		meshes.append(model)

	return meshes
"""

class WatchTimer:
	nesting : int = 0
	timers: list = []
	report = ''
	
	def __init__(self, region:str):
		self.region = region
	
	def __enter__(self) -> None:
		self.start_time = getTime()
		self.nesting = WatchTimer.nesting
		WatchTimer.nesting += 1
		for i,t in enumerate(WatchTimer.timers):
			if t.region == self.region:
				WatchTimer.timers[i] = self
				return
		WatchTimer.timers.append(self)

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		WatchTimer.nesting -= 1
		self.message = self.get_message()
		#print(self.message.encode())
	
	def get_message(self):
		#return ('  ' * self.nesting + f'{self.region} : { self.elapsed_ms() :.1f}ms')	
		return ('  ' * self.nesting + f'{self.region} : { self.elapsed_percent() :.0f}%')	
	
	def elapsed_ms(self):
		return (getTime() - self.start_time) * 1000.0

	def elapsed_percent(self):
		ft = RenderContext.frame_time + 0.00001
		return (getTime() - self.start_time) / ft * 100.0
	
	def capture():
		WatchTimer.report = '\n'.join( list(map(
			lambda t: t.message if hasattr(t, 'message') else t.get_message(),
			WatchTimer.timers))
		)
		WatchTimer.timers.clear()
		return WatchTimer.report

	#def display(x, y, size, color):
	#	rl.DrawText(WatchTimer.report.encode(), x, y, size, color)


CUBE_POSITIONS_24 = np.array((
	# +Z (front)
	(-0.5,-0.5,+0.5), (+0.5,-0.5,+0.5), (+0.5,+0.5,+0.5), (-0.5,+0.5,+0.5),
	# -Z (back)
	(+0.5,-0.5,-0.5), (-0.5,-0.5,-0.5), (-0.5,+0.5,-0.5), (+0.5,+0.5,-0.5),
	# +X (right)
	(+0.5,-0.5,+0.5), (+0.5,-0.5,-0.5), (+0.5,+0.5,-0.5), (+0.5,+0.5,+0.5),
	# -X (left)
	(-0.5,-0.5,-0.5), (-0.5,-0.5,+0.5), (-0.5,+0.5,+0.5), (-0.5,+0.5,-0.5),
	# +Y (top)
	(-0.5,+0.5,+0.5), (+0.5,+0.5,+0.5), (+0.5,+0.5,-0.5), (-0.5,+0.5,-0.5),
	# -Y (bottom)
	(-0.5,-0.5,-0.5), (+0.5,-0.5,-0.5), (+0.5,-0.5,+0.5), (-0.5,-0.5,+0.5),
))

CUBE_NORMALS_24 = np.array(
	([ Vec3( ( 0.0, 0.0, 1.0) ) ] * 4) +   # +Z
	([ Vec3( ( 0.0, 0.0,-1.0) ) ] * 4) +   # -Z
	([ Vec3( ( 1.0, 0.0, 0.0) ) ] * 4) +   # +X
	([ Vec3( (-1.0, 0.0, 0.0) ) ] * 4) +   # -X
	([ Vec3( ( 0.0, 1.0, 0.0) ) ] * 4) +   # +Y
	([ Vec3( ( 0.0,-1.0, 0.0) ) ] * 4),    # -Y
	dtype=np.float32
)

CUBE_UVS_24 = np.array([ (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] * 6, dtype=np.float32)

CUBE_INDICES_36 = np.array([
	0, 1, 2,  2, 3, 0,		 # +Z
	4, 5, 6,  6, 7, 4,		 # -Z
	8, 9, 10,  10, 11, 8,	 # +X
	12, 13, 14,  14, 15, 12, # -X
	16, 17, 18,  18, 19, 16, # +Y
	20, 21, 22,  22, 23, 20, # -Y
], dtype=np.uint32)

RenderContext.resources["cube"] = lambda : (
	Mesh(interleave_mesh_position_normal_uv(CUBE_POSITIONS_24, CUBE_NORMALS_24, CUBE_UVS_24), CUBE_INDICES_36)
)


instance_dtype = np.dtype([
	#("uModel", Mat4_type),
	("iPosition", Vec3_f32_type),
	("iRotation", Vec4_f16_type), # quaternion
	("iScale",    Vec4_f16_type), # last float16 is unused
	("iTint",     Color_type)
])


def linear_to_srgb(x):
	x = np.asarray(x, dtype=np.float32)
	return np.where(
		x <= 0.0031308,
		x * 12.92,
		1.055 * np.power(x, 1.0 / 2.4) - 0.055,
	)

_RGBA_SHIFT = np.array([0, 8, 16, 24], dtype=np.uint32)
def pack_rgba8_srgb(rgba:Vec4):
	rgba = np.array([ *linear_to_srgb(rgba.xyz), rgba.w])
	rgba8 = np.rint(rgba * 255).astype(np.uint32)
	#print(rgba8)
	return np.uint32(np.sum(rgba8 << _RGBA_SHIFT))


def flush_cubes(rp:RenderPass, shader:_Shader, uniformBuffer:_UniformBuffer):
	cube_mesh = RenderContext.resources["cube"]
	
	# draw
	cube_mesh.instance_buffer.upload()
	cube_mesh.draw(rp, shader, uniformBuffer)

	# clear
	cube_mesh.instance_buffer.clear(rp)
	# NOTE: we keep reallocating numpy arrays on cpu
	# optimally we'd reuse them but that complicates implementation
	cube_mesh.instance_buffer.content = np.empty(0,instance_dtype)


def draw_cube(position:Vec3, size:Vec3, color:Vec4, rotation:Quaternion = Quaternion()) -> None:
	instance_data = np.empty(1, instance_dtype)
	instance_data[0]["iPosition"] = position
	instance_data[0]["iRotation"] = rotation
	instance_data[0]["iScale"] = [*size, 0.0] # this is a vec3, shader expects a vec4 so we pad the numpy array
	instance_data[0]["iTint"] = pack_rgba8_srgb(Vec4(color))
	draw_cubes(instance_data)

def draw_cubes(instance_data:np.ndarray) -> None:
	RenderContext.resources["cube"].add_instances(instance_data)

"""
import timeit
import math
import collections

def test(f):
	for i in range(1, 1000001): f(i)

def timetest(f):
	print('{}: {}'.format(timeit.timeit(lambda: test(f), number=10), f.__name__))
"""

def setup_gc_monitor() -> None:
	print(f'{gc.get_threshold()=}')
	#gc.set_threshold(1000, 30, 2) # deflt 2000, 10, 10
	_gc_start = None

	def gc_probe(phase:str, info:dict) -> None:
		nonlocal _gc_start
		if phase == "start":
			_gc_start = getTime()
		elif phase == "stop":
			dt_ms = (getTime() - _gc_start) * 1000
			if not any(info.values()): return
			print(
				f"GC gen={info['generation']} "
				f"collected={info['collected']} "
				f"uncollectable={info['uncollectable']} "
				f"time={dt_ms:.3f} ms"
			)

	gc.callbacks.append(gc_probe)

def setup_memory_monitor() -> Callable:
	import psutil # NOTE: this is not a built-in lib
	process = psutil.Process(os.getpid())

	import tracemalloc
	tracemalloc.start()

	def print_report() -> None:
		rss = process.memory_info().rss
		print(f"process RSS: {rss / 1024 / 1024:.1f} MiB")
		current, peak = tracemalloc.get_traced_memory()
		print(f"traced memory: {current / 1024 / 1024:.2f} MiB")

	return print_report

