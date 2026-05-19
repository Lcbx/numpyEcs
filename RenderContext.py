


import glfw, atexit, time, os, re, gc

import wgpu
from wgpu.utils.glfw_present_info import get_glfw_present_info

import numpy as np

from glob import glob
from typing import Any, Sequence, Iterable, List, Dict, Tuple, NamedTuple, Callable
from jinja2 import Environment, FileSystemLoader, StrictUndefined

getTime = time.perf_counter
sleep = time.sleep


class RenderContext:
	# simple window loop on top of glfw
	# inspired by https://github.com/pygfx/rendercanvas/blob/main/rendercanvas/glfw.py

	def __init__(self):
		self.canvas : wgpu.GPUCanvasContext|None = None
		self.adapter : wgpu.GPUAdapter
		self.device : wgpu.GPUDevice
		self.presentation_format : str

		self.window : glfw._GLFWwindow
		self.windowDimensions : tuple[float,float] = (0.0,0.0)
		self.aspect : float = 1.0

		self.target_frame_time : float|None = None
		self.frame_start : float			   = 0.0
		self.frame_time : float			   = 0.0

		self.depth_format : str = wgpu.TextureFormat.depth24plus
		self.depth :  wgpu.GPUTextureView

		# event_name:[handlers]
		# handlers return True if no need to propagate further
		# NOTE: we should in reverse order if more recent handler have priority
		self.event_handlers : Dict[str, List[Callable]] = {}

		# before startup -> resource_name:func
		# after startup  -> resource_name:resource
		# used to setup utils if imported
		self.resources = {}

	def InitWindow(cls, w:float, h:float, title:str, highpower_gpu:bool=True, target_fps:int=0) -> None:
		""" init a window and wgpu rendering context.
		:param highpower_gpu: use the high performance gpu if there are multiple
		:param target_fps: -1 is limitless, 0 is vsync, other values is fps limit
		"""

		#print(f'{glfw.__version__=}')
		glfw.init()
		atexit.register(cls.cleanup)
		
		# in case monitor setup becomes relevant
		#monitors = glfw.get_monitors()
		#for monitor in monitors:
		#	video_mode = glfw.get_video_mode(monitor)
		#	print(f'{video_mode=}')

		# uncomment to set to fullscreen
		# glfw.set_window_monitor(monitor)

		glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
		#glfw.window_hint(glfw.RESIZABLE, True)
		#glfw.window_hint(glfw.DECORATED, False)

		monitor = glfw.get_primary_monitor()
		monitor_x, monitor_y, monitor_w, monitor_h = glfw.get_monitor_workarea(monitor)
		glfw.window_hint(glfw.POSITION_X, monitor_w-w)
		glfw.window_hint(glfw.POSITION_Y, 30) 

		# had weird case where vsync was not blocking
		if target_fps == 0:
			video_mode = glfw.get_video_mode(monitor)
			#print(f'{video_mode=}')
			target_fps = video_mode.refresh_rate

		if target_fps != -1:
			cls.target_frame_time = 1.0 / float(target_fps)

		# width, height, title, monitor (for full screen), window (for opengl context sharing)
		cls.window = glfw.create_window(w, h, title, None, None)
		cls.setup_callbacks()
		#glfw.set_window_pos(cls.window, monitor_w-w, 30)

		cls.setup_graphics(vsync=target_fps==0,highpower_gpu=highpower_gpu)

		for name, init in cls.resources.items():
			cls.resources[name] = init()

		cls.frame_start = getTime()

	def setup_graphics(cls, *, vsync:bool, highpower_gpu:bool) -> None:
		""" setup gpu compute & render surface """
		present_info = get_glfw_present_info(cls.window, vsync=vsync)
		cls.canvas = wgpu.gpu.get_canvas_context(present_info)
		#print(f'{present_info=}')
		cls.setup_graphics_backend(highpower_gpu=highpower_gpu)
		cls.presentation_format = cls.canvas.get_preferred_format(cls.adapter)
		#print(f'{cls.presentation_format=}')
		#print(f'{cls.canvas._get_capabilities_screen(cls.adapter)=}')

		# usage = wgpu.TextureUsage.RENDER_ATTACHMENT for offsceen
		cls.canvas.configure(device=cls.device, format=cls.presentation_format)

		# ['Fifo', 'FifoRelaxed', 'Mailbox', 'Immediate']
		#cls.canvas.set_present_mode('FifoRelaxed') #edited the wgpu library to add this method
		#cls.canvas._configure_screen_real()

	def setup_graphics_backend(cls, *, highpower_gpu:bool) -> None:
		""" setup gpu compute, not necessarily with canvas output. used notably for tests """
		request_params = {'power_preference': 'high-performance' if highpower_gpu else 'low-power' }
		if cls.canvas: request_params['canvas'] = cls.canvas
		cls.adapter = wgpu.gpu.request_adapter_sync(**request_params)
		#print(f'{cls.adapter.info=}')
		cls.device = cls.adapter.request_device_sync()


	def setup_callbacks(cls) -> None:

		# key is key down, key up
		# char is resulting utf8 character
		# ex: pressing shift + 0 up and down generates ')' char
		glfw.set_key_callback(cls.window, cls.setup_event('key'))
		glfw.set_char_callback(cls.window, cls.setup_event('char'))

		# mouse stuff
		glfw.set_mouse_button_callback(cls.window, cls.setup_event('mouse_click'))
		glfw.set_cursor_pos_callback(cls.window, cls.setup_event('mouse_move'))
		glfw.set_cursor_enter_callback(cls.window, cls.setup_event('mouse_enter'))
		glfw.set_scroll_callback(cls.window, cls.setup_event('mouse_scroll'))#, info_log=True))


	def updateWindowSize(cls, wh : Tuple[int, int]) -> None:
		# NOTE: some versions of glfw send resize events
		w, h = wh
		cls.windowDimensions = wh
		cls.canvas.set_physical_size(w, h)
		cls.aspect = w/h
		cls.depth = cls.device.create_texture(
			size=(w, h, 1),
			format= cls.depth_format,
			usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
		).create_view()


	def capture_mouse(cls, capture:bool=True) -> None:
		""" hide and keep the mouse inside the window
		NOTE: there's also glfw.CURSOR_HIDDEN which lets the cursor get out of the window
		you can create cursors ex: cursor = glfw.create_cursor(image, 0, 0) # generating image is some work though
		use standard cursors ex: cursor = glfw.create_standard_cursor(cursor_flag)
		then glfw.set_cursor(self._window, cursor)
		"""
		glfw.set_input_mode(cls.window, glfw.CURSOR, glfw.CURSOR_DISABLED if capture else glfw.CURSOR_NORMAL)


	def subscribe_event(cls, channel_name:str, handler:Callable) -> None:
		cls.event_handlers[channel_name].append(handler)

	def setup_event(cls, channel_name:str, mapper:Callable|None=None, info_log:bool=False) -> Callable:
		
		def print_handler(*args):
			print(channel_name, *args)
		
		handlers = []
		if info_log:
			handlers.append(print_handler)

		cls.event_handlers[channel_name] = handlers

		# usually first arg is window and we don't need that
		def deflt_mapper(*args)->None:
			return args[1:]

		mapper = mapper if mapper else deflt_mapper

		def handlers_call(*args)->None:
			if mapper: args = mapper(*args)
			for handler in cls.event_handlers[channel_name]:
				if handler(*args): break

		return handlers_call


	def cleanup(cls):

		cls.canvas.unconfigure()

		glfw.destroy_window(cls.window)

		# work around https://github.com/glfw/glfw/issues/1766
		end_time = time.perf_counter() + 0.1
		while time.perf_counter() < end_time:
			glfw.wait_events_timeout(end_time - time.perf_counter())

		glfw.terminate()


	def WindowLoop(cls) -> bool:

		wh = glfw.get_framebuffer_size(cls.window)
		if wh != cls.windowDimensions and wh[0]>0 and wh[1]>0:
			cls.updateWindowSize(wh)

		cls.canvas.present()
		
		# save frame start timestamp, frame time, cap fps
		cls._frame_pacing()

		glfw.poll_events()

		# TODO: implement a strategy to ensure gc does not freeze the app
		#gc.collect(0) # not a good strategy

		return not glfw.window_should_close(cls.window)

	_FRAME_WAIT_MARGIN = 0.001 # 1ms
	def _frame_pacing(cls):
		now = getTime()
		cls.frame_time = now - cls.frame_start
		
		# if we have a target frame time, wait until we m1atch it
		if cls.target_frame_time and cls.frame_time < cls.target_frame_time:
			# margin is because scheduled sleep is not guaranteed to end on time
			sleep_time = cls.target_frame_time - cls.frame_time - cls._FRAME_WAIT_MARGIN
			if sleep_time > 0.0: sleep(sleep_time)
			# busy wait for the last 1ms
			wait_end = cls.frame_start + cls.target_frame_time
			while now < wait_end: now = getTime()
		
		cls.frame_start = now

	def Command(cls) -> wgpu.CommandEncoder:
		return cls.device.create_command_encoder()

	def RenderPass(cls, *args, **kwargs) -> '_RenderPass':
		return _RenderPass(*args, **kwargs)

	def Shader(cls, *args, **kwargs) -> '_Shader':
		return _Shader(*args, **kwargs)

RenderContext = RenderContext()

class _RenderPass:

	def __init__(self,
			#shader:BetterShader = None,
			camera:Camera|None = None,
			#texture:img.Framebuffer = None
			clear_color:tuple=(0.0, 0.0, 0.0, 1.0)
		):
		#self.shader = shader
		self.camera = camera
		self.clear_color : Tuple[float,float,float,float] = clear_color
		#self.texture = texture
		self.render_pass : wgpu.GPURenderCommandsMixin = None
		self.commands : List[wgpu.CommandEncoder] = None

	def __enter__(self):
		# TODO: support drawing into a custom framebuffer (self.texture)
		#if self.shader: self.shader.bind()
		#if self.texture: self.texture.bind()
		if self.camera:
			self.view = self.camera.view()
			# TODO: determine aspect from texture dimensions
			aspect = RenderContext.aspect
			self.projection = self.camera.projection(aspect)
		#	if 'uView' in self.shader._uniforms: self.shader['uView'] = self.view
		#	if 'uProj' in self.shader._uniforms: self.shader['uProj'] = self.projection
		#	if 'uViewProj' in self.shader._uniforms: self.shader['uViewProj'] = self.view @ self.projection
		command = RenderContext.Command()
		self.render_pass = command.begin_render_pass(
			color_attachments=[
				wgpu.RenderPassColorAttachment(
					view=RenderContext.canvas.get_current_texture().create_view(),
					clear_value=self.clear_color,
					load_op="clear",
					store_op="store",
				)
			],
			depth_stencil_attachment=
				wgpu.RenderPassDepthStencilAttachment(
					view = RenderContext.depth,
					depth_clear_value = 1.0,
					depth_load_op= "clear",
					depth_store_op= "store",
				)
		)
		self.commands = [command]
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		self.render_pass.end()
		RenderContext.device.queue.submit(
			[ command.finish() for command in self.commands ]
		)



_GL_to_dtype: Dict[str, np.dtype] = {
	"float":	 np.dtype( (np.float32, () )   ),
	"vec2": 	 np.dtype( (np.float32, (2,))  ),
	"vec3": 	 np.dtype( (np.float32, (3,))  ),
	"vec4": 	 np.dtype( (np.float32, (4,))  ),
	"mat4": 	 np.dtype( (np.float32, (4,4)) ),
	"uint": 	 np.dtype( (np.uint32,  () )   ),
	"uvec2":	 np.dtype( (np.int32,   (2,))  ),
	"uvec3":	 np.dtype( (np.int32,   (3,))  ),
	"uvec4":	 np.dtype( (np.int32,   (4,))  ),
	"int":  	 np.dtype( (np.int32,   () )   ),
	"ivec2":	 np.dtype( (np.int32,   (2,))  ),
	"ivec3":	 np.dtype( (np.int32,   (3,))  ),
	"ivec4":	 np.dtype( (np.int32,   (4,))  ),
	"sampler2D": np.dtype( (np.uint32,  () )   ), 
}

def glsl_to_dtype(spec:str)-> np.dtype:

	# array case
	if '[' in spec:
		definition = spec.replace(']', '').split('[')
		spec = definition[0]
		counts = tuple(map(int,definition[1:]))
	else: counts = None

	info = _GL_to_dtype.get(spec)
	if info is None:
		raise ValueError(f"Unsupported type {spec!r}")

	if counts:
		# rebuild item dimensions
		info = np.dtype( (info.base, info.shape + counts) )
	
	return info


def make_std430_dtype( original: List[Tuple[str, np.dtype]] ) -> np.dtype:

	dfields: List[Tuple[str, np.dtype]] = []
	offset = 0
	pad_idx = 0

	for name, dtype in original:
		#print(name, dtype)
		dtype = np.dtype(dtype)

		dfields.append((name, dtype))
		offset += dtype.itemsize

		#print(dfields)
		missing_offset = (16 - offset) % 16
		if missing_offset != 0:
			print(f'adding new pad {missing_offset}')
			pad_type = None
			pad_count = 0 
			if missing_offset % 4 == 0:
				pad_type = "u4"
				pad_count = missing_offset//4
			elif missing_offset % 2 == 0:
				pad_type = "u2"
				pad_count = missing_offset//2
			else:
				pad_type = "u1"
				pad_count = missing_offset
			dfields.append( (f"_pad{pad_idx}", np.dtype( (pad_type, pad_count) )) ) 
			offset += missing_offset
			pad_idx += 1

	#print(dfields)

	#return np.dtype(dfields, align=False)
	return np.dtype(dfields, align=True)



_VertexFormat : Dict[Any, Tuple[str, int]] = {
	np.int8:	("sint8",   1),
	np.uint8:	("uint8",   1),
	np.int16:	("sint16",  2),
	np.uint16:	("uint16",  2),
	np.int32:	("sint32",  4),
	np.uint32:	("uint32",  4),
	np.uint32:	("uint32",  4),
	np.float16:	("float16", 2),
	np.float32:	("float32", 4),
	np.float64:	("float64", 8),
}
# NOTE: wgpu.vertexFormat enum are strings in wgpu-py 
def dtype_to_vertex_format(dtype: np.dtype) -> List | str:

	dtype = np.dtype(dtype)

	if dtype.fields is not None:
		fields_data = dtype.fields.items()
		#print(fields_data)
		res = []
		for name, (field_dtype, offset) in fields_data:
			#print(name, field_dtype)
			vFormat = dtype_to_vertex_format(field_dtype)
			#print(name, field_dtype, vFormat)
			if isinstance(vFormat,str):
				res.append( (vFormat, offset) )
			elif isinstance(vFormat,List):
				for vf, local_offset in vFormat:
					res.append( (vf, offset+local_offset) )
		return res

	# Scalar case
	if dtype.subdtype is None:
		try:
			(prefix, scalar_size) = _VertexFormat[dtype]
		except KeyError:
			raise TypeError(f"Unsupported vertex scalar dtype: {dtype}")

		return getattr(wgpu.VertexFormat, prefix)

	# Array case
	base_dtype, shape = dtype.subdtype
	base_type = base_dtype.type

	count = shape[0]

	if count < 1 or count > 4:
		raise ValueError(f"VertexFormat arrays must have 1–4 components (found {count} for {dtype.subdtype}")
	try:
		(prefix, scalar_size) = _VertexFormat[base_type]
	except KeyError:
		raise TypeError(f"Unsupported vertex base dtype: {base_type}")

	shape_len = len(shape)
	vFormat = (
			 getattr(wgpu.VertexFormat, prefix) if count == 1
		else getattr(wgpu.VertexFormat, f"{prefix}x{count}")
	)

	if shape_len == 1:
		return vFormat
	# matrix 
	if shape_len == 2:
		row_count = shape[1]
		return [ (vFormat, scalar_size * count * i) for i in range(row_count) ]
	
	else:
		raise TypeError(f"Only 1D or 2D array dtypes are valid vertex attributes (found {shape} for {dtype.subdtype})")


# TODO: don't compile shaders we already compiled before
class _Shader:


	def __init__(self, *args, **kwargs):
		device = RenderContext.device
		source = self.ShaderSource(*args, **kwargs)
		self.source = source

		#print(source.vertex_glsl)
		#print(source.fragment_glsl)

		try:
			self.vert_module = device.create_shader_module(label="shader.vert",code=source.vertex_glsl)
		except Exception as e:
			print(source.vertex_glsl)
			raise e
		try:
			self.frag_module = device.create_shader_module(label="shader.frag",code=source.fragment_glsl)
		except Exception as e:
			print(source.fragment_glsl)
			raise e

		self.uniforms_dtype = make_std430_dtype(
			[(name, glsl_to_dtype(spec)) for spec, name in self.source.uniforms]
		)

		self.bind_group_layout = device.create_bind_group_layout(
			entries=[
				wgpu.BindGroupLayoutEntry(
					binding=0,
					visibility=wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
					buffer=wgpu.BufferBindingLayout(
						type=wgpu.BufferBindingType.uniform,
						has_dynamic_offset=False,
						#min_binding_size=self.uniforms_dtype.itemsize,
					),
				)
			]
		)		

		self.pipeline_layout = device.create_pipeline_layout(
			bind_group_layouts=[self.bind_group_layout]
		)

	# NOTE: we pass the shader since the mesh needs it for setup
	def UniformBuffer(self, *args, **kwargs) -> '_UniformBuffer':
		return _UniformBuffer(self, *args, **kwargs)

	def ShaderSource(self, *args, **kwargs) -> 'ShaderSource':
		return ShaderSource(*args, **kwargs)

	def create_render_pipeline(self, buffers_spec:List[wgpu.VertexBufferLayout]) -> wgpu.GPURenderPipeline:
		return RenderContext.device.create_render_pipeline(
			layout=self.pipeline_layout,
			#multisample = wgpu.MultisampleState(
				# needs render texture to have the same multisample count
		   		#count = 4,  #u32
				#mask # u64
				#alpha_to_coverage_enabled # bool
			#),
			vertex=wgpu.VertexState(
				module=self.vert_module,
				entry_point="main",
				buffers=buffers_spec,
			),
			primitive=wgpu.PrimitiveState(
				topology=wgpu.PrimitiveTopology.triangle_list,
				cull_mode=wgpu.CullMode.back,
				front_face=wgpu.FrontFace.ccw,
			),
			depth_stencil=wgpu.DepthStencilState(
				format=RenderContext.depth_format,
				depth_write_enabled=True,
				depth_compare=wgpu.CompareFunction.less,
			),
			fragment=wgpu.FragmentState(
				module=self.frag_module,
				entry_point="main",
				targets=[
					wgpu.ColorTargetState(format=RenderContext.presentation_format)
				],
			),
		)

def nearest_pow2(n: int) -> int:
	# NOTE: bit shift alone is actually faster
	#if n & (nm1:=n-1) == 0: return n
	return 1<<(n-1).bit_length()

# NOTE: permanent buffers need to mark dirty ranges and upload changes only
# irl we have a lot to gain from storing the positions of moving geometry in different buffers
class GpuBuffer:

	def __init__(self, content : np.ndarray, usage:wgpu.BufferUsage):
		self.content = content
		self.handle = RenderContext.device.create_buffer_with_data(data=content, usage=usage)
		#self.handle = RenderContext.device.create_buffer(size=content.nbytes, usage=usage)

	def resize(self, count:int):
		#print('count', count, self.content.size)
		size = nearest_pow2(count * self.content.itemsize)
		current = self.handle.size
		if size > current:
			#print('resize', size, current)
			self.handle = RenderContext.device.create_buffer(size=size, usage=self.handle.usage)

	def upload(self) -> None:
		RenderContext.device.queue.write_buffer(self.handle, 0, self.content)

	def clear(self, rp:RenderPass) -> None:
		command = RenderContext.Command()
		command.clear_buffer(self.handle)
		rp.commands.append(command)



class _UniformBuffer(GpuBuffer):

	def __init__(self, shader : _Shader):
		device = RenderContext.device
		self.shader = shader

		content = np.empty((1),dtype=shader.uniforms_dtype)

		super().__init__(content, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

		self.bind_group = device.create_bind_group(
			layout=shader.bind_group_layout,
			entries=[
				wgpu.BindGroupEntry(
					binding=0,
					resource=wgpu.BufferBinding(
						buffer=self.handle,
						offset=0,
						size=content.nbytes,
					),
				)
			],
		)



# TODO: pool meshes with same vertex attributes
class Mesh:

	# NOTE: is there a point to supporting meshes with no indices these days
	def __init__(self, vertex_data:np.ndarray, indices:np.ndarray):
		device = RenderContext.device

		self.index_count = indices.size
		vertex_dtype = vertex_data.dtype

		vformat = dtype_to_vertex_format(vertex_dtype)
		vertex_loc_count = len(vformat) 
		#print("vertex", vformat, vertex_dtype.itemsize)
		self.vertexAttributes = [ wgpu.VertexAttribute(
				shader_location = i,
				offset=offset,
				format=vertexFormat
			) for i, (vertexFormat, offset)
			in enumerate(vformat)
		]
		
		self.vertex_buffer = GpuBuffer(vertex_data, wgpu.BufferUsage.VERTEX)
		self.index_buffer = GpuBuffer(indices, wgpu.BufferUsage.INDEX)
		
		self.instance_buffer : GpuBuffer|None = None
		self.instance_count = 1

		self.buffers_spec = [
			wgpu.VertexBufferLayout(
				array_stride=vertex_dtype.itemsize,
				step_mode=wgpu.VertexStepMode.vertex,
				attributes=self.vertexAttributes
			)
		]

	def add_instances(self, instance_data:np.ndarray) -> None:
		if buffer := self.instance_buffer:
			instance_data = np.append(buffer.content, instance_data)
		self.set_instances(instance_data)

	def set_instances(self, instance_data:np.ndarray) -> None:
		
		self.instance_count = instance_data.size

		# NOTE: if the buffer exists we don't upload it immediately
		if buffer := self.instance_buffer:
			buffer.resize(self.instance_count)
			buffer.content = instance_data
			return

		instance_dtype = instance_data.dtype
		vformat = dtype_to_vertex_format(instance_dtype)
		#print("instance", vformat, instance_dtype.itemsize)
		instanceAttributes = [ wgpu.VertexAttribute(
				shader_location = i + len(self.vertexAttributes),
				offset=offset,
				format=vertexFormat
			) for i, (vertexFormat, offset)
			in enumerate(vformat)
		]

		self.instance_buffer = GpuBuffer(instance_data, wgpu.BufferUsage.VERTEX|wgpu.BufferUsage.COPY_DST)

		self.buffers_spec.append(
			wgpu.VertexBufferLayout(
				array_stride=instance_dtype.itemsize,
				step_mode=wgpu.VertexStepMode.instance,
				attributes=instanceAttributes
			))


	def draw(self, renderpass:RenderPass, shader:'_Shader', uniforms:'_UniformBuffer') -> None:
		render_pass = renderpass.render_pass
		render_pass.set_pipeline(shader.create_render_pipeline(self.buffers_spec))
		render_pass.set_bind_group(0, uniforms.bind_group)
		render_pass.set_vertex_buffer(0, self.vertex_buffer.handle)
		if self.instance_buffer:
			render_pass.set_vertex_buffer(1, self.instance_buffer.handle)
		render_pass.set_index_buffer(self.index_buffer.handle, wgpu.IndexFormat.uint32)

		#index_count (int) – The number of indices to draw.
		#instance_count (int) – The number of instances to draw. Default 1.
		#first_index (int) – The index offset. Default 0.
		#base_vertex (int) – A number added to each index in the index buffer. Default 0.
		#first_instance (int) – The instance offset. Default 0.
		render_pass.draw_indexed(self.index_count, self.instance_count, 0, 0, 0)


def build_shader_program(shaderPath : str, **kwargs) -> Tuple['ShaderSource', _Shader]:
	sh = RenderContext.Shader(filepath=shaderPath, **kwargs)
	return sh.source, sh


class DefaultFalseDict(Dict):
	# Dict that returns False for any missing key; used in feature resolution
	def __missing__(self, key):
		return False

# Parses a shader definition file containing two functions: vertex() and fragment().
# Extracts uniforms, varying, in, and out variables, then generates GLSL code for both stages.
class ShaderSource:

	# Regex to capture qualifier, type, and name from declarations
	_decl_pattern = re.compile(
		r"^(?:layout\(location\s*=\s*(\d+)\)\s+)?"
		r"(uniform|in|out|(?:flat\s+)?varying|const)\s+"
		r"(\w+(?:\s*\[\s*\d*\s*\])*)\s+"
		r"([^;]+).*?;",
		re.MULTILINE
	)

	# Regex to extract function definitions with bodies
	_func_pattern = re.compile(
		r'\n'						# start at a newline
		r'[\w\*\s&<>]+?\s+'			# return type (e.g. void, bool, vec4, const mat4&)
		r'([A-Za-z_]\w*)'			# function name
		r'\s*\(([^)]*)\)\s*'		# argument list
		r'(?:\{\n|\n\{\n)'			# opening brace on same or next line
		r'([\s\S]*?)'				# function body (non-greedy)
		r'\n\}'						# closing brace at column 0
	)

	def create_variable_regex(self, name:str) -> re.Pattern:
		escaped = re.escape(name)
		_GLSL_IDENT_CHAR = "[A-Za-z0-9_]"
		pattern = rf"(?<!{_GLSL_IDENT_CHAR}){escaped}(?!{_GLSL_IDENT_CHAR})"
		return re.compile(pattern)

	_vertex_start = 'void vertex()'
	_fragment_start = 'void fragment()'
	_main_start = 'void main()'

	def __init__(
		self,
		*,
		source       : str            | None = None,
		filepath     : str            | None = None,
		basedir      : str            | None = None,
		features     : Sequence[str]  | None = None,
		params       : Dict[str, Any] | None = None,
		glsl_version : str = '#version 450 core'
	):
		# :param source: master shader definition code (template).
		# :param filepath: Path to the master shader definition file (template).
		# :param features: Dict of feature flags for conditionals, e.g. {'FEATURE_FOG': True}.
		# :param params: Any extra variables you want available in templates.
		# :param glsl_version: Override #version.

		assert( source or filepath )

		self.source = source if source else open(filepath, 'r').read()
		self._basedir = (
			basedir if basedir
			else os.path.dirname(os.path.abspath(filepath)) if filepath
			else glob('**/shaders/', recursive=True)
		)
		self._glsl_version  = glsl_version
		
		self.uniforms      = [] # list[(type, name)]
		self.varyings      = [] # list[(type, name)]
		self.ins           = [] # list[(loc, type, name)]
		self.outs          = [] # list[(loc, type, name)]
		self.consts        = [] # list[(type, name)]
		self.functions     = [] # list[str]
		self.vertex_glsl   = '' # rendered vertex GLSL
		self.fragment_glsl = '' # rendered fragment GLSL

		#print(f'{self._basedir=}')
		#print(f'compiling {filepath}')

		# Render the shader source through Jinja2 (handles #if / #include)
		text = self._render_template(features, params)

		# Parse the rendered text for declarations & functions
		self._parse_rendered_text(text)

		# Generate final vertex/fragment GLSL
		self._generate_glsl()

	def _render_template(self, features: Sequence[str]|None, params: Dict[str, Any]|None) -> str:
		# evaluate the preprocessor sections using jinja2

		env = Environment(
			loader=FileSystemLoader(self._basedir),
			undefined=StrictUndefined,  # fail fast for missing vars
			autoescape=False,			# GLSL is not HTML
			keep_trailing_newline=True,
			trim_blocks=True,
			lstrip_blocks=True,
			line_statement_prefix='#' # << key: enable #if/#endif/#include
		)
		template = env.from_string(self.source)

		featuresDict = DefaultFalseDict()
		if features is not None: featuresDict.update({k: True for k in features})

		return template.render(FEATURES=featuresDict, PARAMS=params or {})

	
	def _parse_rendered_text(self, text: str) -> None:
		# Declarations
		for loc, qual, typ, name in self._decl_pattern.findall(text):
			if qual   == 'uniform':
				self.uniforms.append((typ, name))
			elif qual == 'varying':
				self.varyings.append((loc, typ, name, False))
			elif qual == 'flat varying':
				self.varyings.append((loc, typ, name, True))
			elif qual == 'in':
				self.ins.append((loc, typ, name))
			elif qual == 'out':
				self.outs.append((loc, typ, name))
			elif qual == 'const':
				self.consts.append((typ, name))

		# Functions (full match = whole function; we keep their text)
		self.functions = [m.group(0).strip() for m in self._func_pattern.finditer(text)]

		#print(*[ f[0:20] for f in self.functions], sep='\n')

		# Find vertex()/fragment() bodies
		for i, f in enumerate(self.functions):
			if f.startswith(self._vertex_start):
				self._vertex_body = i
				#print('-> got vert', hasattr(self, '_vertex_body'))
			elif f.startswith(self._fragment_start):
				self._fragment_body = i
				#print('-> got frag', hasattr(self, '_fragment_start'))


	def _generate_glsl(self) -> None:
		def inoutFmt(loc, tup, *, inoutStr):
			if len(tup) == 4:
				loc_, typ, name, flat = tup
			else:
				loc_, typ, name = tup
				flat = False
			loc = loc_ if loc_ else loc
			return f'layout(location = {loc}){(' flat' if flat else '')} {inoutStr} {typ} {name};'
		def inStr(entry):
			return inoutFmt(*entry, inoutStr='in')
		def outStr(entry):
			return inoutFmt(*entry, inoutStr='out')

		# workaround for simple uniforms : make one big uniform struct and use aliases
		uniforms_definition = f'layout(set=0, binding=0, std430) uniform Uniforms {{\n {(
				"".join(f"\t{typ} {name};\n" for typ, name in self.uniforms)
			)} }} _U;'

		# replace uniform references by alias
		for i, f in enumerate(self.functions):
			for _, name  in self.uniforms:
				rx = self.create_variable_regex(name)
				self.functions[i] = rx.sub(f'_U.{name}', f)

		v_lines = (
			self._glsl_version, '',
			*map(inStr, enumerate(self.ins)), '',
			uniforms_definition, '',
			*map(outStr, enumerate(self.varyings)), '',
			*map(lambda kv: f'const {kv[0]} {kv[1]};', self.consts), '',
			*self.functions[:self._vertex_body], '',
			self.functions[self._vertex_body].replace(self._vertex_start, self._main_start)
		)
		self.vertex_glsl = '\n'.join(v_lines)

		# Helper: functions after vertex() but not fragment()
		start_idx = (self._vertex_body + 1) if self._vertex_body is not None else 0
		functions_after_vertex_but_not_fragment = [
			self.functions[i] for i in range(start_idx, len(self.functions))
			if i != self._fragment_body
		]

		# Fragment shader (required)
		f_lines = [
			self._glsl_version, '',
			*map(inStr, enumerate(self.varyings)), '',
			uniforms_definition, '',
			*map(outStr, enumerate(self.outs)), '',
			*map(lambda kv: f'const {kv[0]} {kv[1]};', self.consts), '',
			*functions_after_vertex_but_not_fragment, '',
			self.functions[self._fragment_body].replace(self._fragment_start, self._main_start)
		]
		self.fragment_glsl = '\n'.join(f_lines)



"""

def GenTextureMipmaps(texture : img.Texture):
	#gl.glBindTexture(gl.GL_TEXTURE_2D, texture.id)
	#gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
	#gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
	raise Exception("not implemented")

# NOTE: putting a smaller texture into a bigger one is not allowed
def TransferDepth(from_fbo:int, f_w:int, f_h:int, to_fbo:int, t_w:int, t_h:int):
	#gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, from_fbo)
	#gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, to_fbo)
	#gl.glBlitFramebuffer(
	#	0, 0, f_w, f_h,
	#	0, 0, t_w, t_h,
	#	#gl.GL_COLOR_BUFFER_BIT,
	#	gl.GL_DEPTH_BUFFER_BIT,
	#	gl.GL_NEAREST
	#)
	raise Exception("not implemented")


def SetPolygonOffset(value:float, flat:float=0.0):
	#gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
	#gl.glPolygonOffset(value, flat)
	raise Exception("not implemented")
def DisablePolygonOffset():
	#gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
	raise Exception("not implemented")

def DrawTexture(tex:img.Texture, width:float, height:float):
	# screen-wide rectangle, y-flipped due to default OpenGL coordinates
	#rl.DrawTextureRec(tex, (0, 0, width, -height), (0, 0), rl.WHITE)
	raise Exception("not implemented")


def create_frame_buffer(width:int, height:int,
						colorFormat = gl.GL_RGBA8,
						depth_map:bool = False,
						samples:int = 4):

	#framebuffer = img.Framebuffer()

	#create_tex = create_simple_texture if samples==1 else TextureMsaa.create

	#if colorFormat:
	#	color_tex = create_tex(width=width, height=height, internalformat=colorFormat, samples=samples)
	#	framebuffer.attach_texture(color_tex, attachment=gl.GL_COLOR_ATTACHMENT0)

	#if depth_map:
	#	depth_tex = create_tex(width=width, height=height, internalformat=gl.GL_DEPTH_COMPONENT24, fmt=gl.GL_DEPTH_COMPONENT, samples=samples)
	#	framebuffer.attach_texture(depth_tex, attachment=gl.GL_DEPTH_ATTACHMENT)
	#	depth = depth_tex
	#else:
	#	depth_rb = img.buffer.Renderbuffer.create(width, height, gl.GL_DEPTH_COMPONENT24, samples=samples)
	#	framebuffer.attach_renderbuffer(depth_rb, attachment=gl.GL_DEPTH_ATTACHMENT)
	#	depth = depth_rb
	#return framebuffer
	raise Exception("not implemented")

"""