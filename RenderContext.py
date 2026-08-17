from __future__ import annotations

from common import *
import time, os, re
from threading import Lock
from dataclasses import dataclass

import wgpu
from wgpu.utils.glfw_present_info import get_glfw_present_info
from wgpu import BufferUsage

import numpy as np

from glob import glob
from jinja2 import Environment, FileSystemLoader, StrictUndefined


getTime = time.perf_counter
sleep = time.sleep
Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class ColorAttachment:
	view: wgpu.GPUTextureView
	format: str
	clear: Color | None = None
	store: bool = True

	def descriptor(self) -> wgpu.RenderPassColorAttachment:
		return wgpu.RenderPassColorAttachment(
			view=self.view,
			clear_value=self.clear or (0.0, 0.0, 0.0, 0.0),
			load_op="clear" if self.clear is not None else "load",
			store_op="store" if self.store else "discard",
		)


@dataclass(frozen=True)
class DepthAttachment:
	view: wgpu.GPUTextureView
	format: str
	clear: float | None = None
	store: bool = True
	read_only: bool = False

	def descriptor(self) -> wgpu.RenderPassDepthStencilAttachment:
		return wgpu.RenderPassDepthStencilAttachment(
			view=self.view,
			depth_clear_value=self.clear or 0.0,
			depth_load_op="clear" if self.clear is not None else "load",
			depth_store_op="store" if self.store else "discard",
			depth_read_only=self.read_only,
		)


class _RenderContext:
	# simple window loop on top of glfw
	# inspired by https://github.com/pygfx/rendercanvas/blob/main/rendercanvas/glfw.py
	def __init__(self):
		self.canvas: wgpu.GPUCanvasContext | None = None
		self.adapter: wgpu.GPUAdapter
		self.device: wgpu.GPUDevice
		self.presentation_format: str

		self.window: glfw._GLFWwindow
		self.windowDimensions: tuple[float, float] = (0.0, 0.0)
		self.aspect: float = 1.0

		self.target_frame_time: float | None = None
		self.frame_start: float = 0.0
		self.frame_time: float = 0.0

		self.depth_format: str = wgpu.TextureFormat.depth24plus
		self.depth: wgpu.GPUTextureView
		self._screen_view: wgpu.GPUTextureView | None = None

		# event_name:[handlers]
		# handlers return True if no need to propagate further
		self.event_handlers: Dict[str, List[Callable]] = {}

		# before startup -> resource_name:func
		# after startup  -> resource_name:resource
		# used to setup utils if imported
		self.resources = {}

	def InitWindow(
		cls,
		w: float,
		h: float,
		title: str,
		highpower_gpu: bool = True,
		target_fps: int = 0,
	) -> None:
		"""Init a window and wgpu rendering context.

		:param highpower_gpu: use the high performance gpu if there are multiple
		:param target_fps: -1 is limitless, 0 is vsync, other values is fps limit
		"""
		global glfw
		import glfw, atexit

		glfw.init()
		atexit.register(cls.cleanup)

		glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

		monitor = glfw.get_primary_monitor()
		monitor_x, monitor_y, monitor_w, monitor_h = glfw.get_monitor_workarea(monitor)
		glfw.window_hint(glfw.POSITION_X, monitor_w - w)
		glfw.window_hint(glfw.POSITION_Y, 30)

		# had weird case where vsync was not blocking
		if target_fps == 0:
			video_mode = glfw.get_video_mode(monitor)
			target_fps = video_mode.refresh_rate

		if target_fps != -1:
			cls.target_frame_time = 1.0 / float(target_fps)

		cls.window = glfw.create_window(w, h, title, None, None)
		cls.setup_callbacks()
		cls.setup_graphics(vsync=target_fps == 0, highpower_gpu=highpower_gpu)

		for name, init in cls.resources.items():
			cls.resources[name] = init()

		cls.frame_start = getTime()

	def setup_graphics(cls, *, vsync: bool, highpower_gpu: bool) -> None:
		"""Setup gpu compute & render surface."""
		present_info = get_glfw_present_info(cls.window, vsync=vsync)
		cls.canvas = wgpu.gpu.get_canvas_context(present_info)
		cls.setup_graphics_backend(highpower_gpu=highpower_gpu)
		cls.presentation_format = cls.canvas.get_preferred_format(cls.adapter)
		cls.canvas.configure(device=cls.device, format=cls.presentation_format)

	def setup_graphics_backend(cls, *, highpower_gpu: bool) -> None:
		"""Setup gpu compute, not necessarily with canvas output. Used notably for tests."""
		request_params = {
			"power_preference": "high-performance" if highpower_gpu else "low-power"
		}
		if cls.canvas:
			request_params["canvas"] = cls.canvas
		cls.adapter = wgpu.gpu.request_adapter_sync(**request_params)
		cls.device = cls.adapter.request_device_sync()

	def setup_callbacks(cls) -> None:
		glfw.set_key_callback(cls.window, cls.setup_event("key"))
		glfw.set_char_callback(cls.window, cls.setup_event("char"))
		glfw.set_mouse_button_callback(cls.window, cls.setup_event("mouse_click"))
		glfw.set_cursor_pos_callback(cls.window, cls.setup_event("mouse_move"))
		glfw.set_cursor_enter_callback(cls.window, cls.setup_event("mouse_enter"))
		glfw.set_scroll_callback(cls.window, cls.setup_event("mouse_scroll"))

	def updateWindowSize(cls, wh: Tuple[int, int]) -> None:
		w, h = wh
		cls.windowDimensions = wh
		cls.canvas.set_physical_size(w, h)
		cls.aspect = w / h
		cls.depth = cls.device.create_texture(
			size=(w, h, 1),
			format=cls.depth_format,
			usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
		).create_view()

	def capture_mouse(cls, capture: bool = True) -> None:
		"""Hide and keep the mouse inside the window."""
		glfw.set_input_mode(
			cls.window,
			glfw.CURSOR,
			glfw.CURSOR_DISABLED if capture else glfw.CURSOR_NORMAL,
		)

	def subscribe_event(cls, channel_name: str, handler: Callable) -> None:
		cls.event_handlers[channel_name].append(handler)

	def setup_event(
		cls,
		channel_name: str,
		mapper: Callable | None = None,
		info_log: bool = False,
	) -> Callable:
		def print_handler(*args):
			print(channel_name, *args)

		handlers = []
		if info_log:
			handlers.append(print_handler)
		cls.event_handlers[channel_name] = handlers

		def deflt_mapper(*args) -> tuple:
			return args[1:]

		mapper = mapper if mapper else deflt_mapper

		def handlers_call(*args) -> None:
			if mapper:
				args = mapper(*args)
			for handler in cls.event_handlers[channel_name]:
				if handler(*args):
					break

		return handlers_call

	def cleanup(cls) -> None:
		if cls.canvas:
			cls.canvas.unconfigure()

		if hasattr(cls, "window"):
			glfw.destroy_window(cls.window)

		# work around https://github.com/glfw/glfw/issues/1766
		end_time = time.perf_counter() + 0.1
		while time.perf_counter() < end_time:
			glfw.wait_events_timeout(end_time - time.perf_counter())
		glfw.terminate()

	def WindowLoop(cls) -> bool:
		"""Present the previous image, update window/input state, and continue the loop.

		This deliberately has no command encoder associated with it. Independent command
		contexts can be created at any point during the iteration.
		"""
		wh = glfw.get_framebuffer_size(cls.window)
		if wh != cls.windowDimensions and wh[0] > 0 and wh[1] > 0:
			cls.updateWindowSize(wh)

		if cls._screen_view is not None:
			cls.canvas.present()
			cls._screen_view = None

		cls._frame_pacing()
		glfw.poll_events()
		return not glfw.window_should_close(cls.window)

	_FRAME_WAIT_MARGIN = 0.001 # 1ms

	def _frame_pacing(cls) -> None:
		now = getTime()
		cls.frame_time = now - cls.frame_start

		if cls.target_frame_time and cls.frame_time < cls.target_frame_time:
			sleep_time = cls.target_frame_time - cls.frame_time - cls._FRAME_WAIT_MARGIN
			if sleep_time > 0.0:
				sleep(sleep_time)
			wait_end = cls.frame_start + cls.target_frame_time
			while now < wait_end:
				now = getTime()

		cls.frame_start = now

	# commands -----------------------------------------------------------------

	def commands(cls, label: str | None = None) -> CommandContext:
		return CommandContext(label=label)

	def Command(cls, label: str | None = None) -> wgpu.GPUCommandEncoder:
		"""Raw command encoder escape hatch."""
		return cls.device.create_command_encoder(label=label or "")

	def submit(cls, *command_buffers: wgpu.GPUCommandBuffer) -> None:
		cls.device.queue.submit(list(command_buffers))

	# attachments ---------------------------------------------------------------

	def screen(
		cls,
		*,
		clear: Color | None = None,
		store: bool = True,
	) -> ColorAttachment:
		if cls._screen_view is None:
			cls._screen_view = cls.canvas.get_current_texture().create_view()
		return ColorAttachment(cls._screen_view, cls.presentation_format, clear, store)

	def depth_attachment(
		cls,
		*,
		clear: float | None = None,
		store: bool = True,
		read_only: bool = False,
	) -> DepthAttachment:
		return DepthAttachment(cls.depth, cls.depth_format, clear, store, read_only)

	# resource factories --------------------------------------------------------

	def Buffer(cls, *args, **kwargs) -> GpuBuffer:
		return GpuBuffer(*args, **kwargs)

	def Texture(cls, *args, **kwargs) -> Texture:
		return Texture(*args, **kwargs)

	def Sampler(cls, *args, **kwargs) -> Sampler:
		return Sampler(*args, **kwargs)

	def Mesh(cls, *args, **kwargs) -> Mesh:
		return Mesh(*args, **kwargs)

	def Shader(cls, *args, **kwargs) -> Shader:
		return Shader(*args, **kwargs)

	def RenderPipeline(cls, *args, **kwargs) -> RenderPipeline:
		return RenderPipeline(*args, **kwargs)

	def ComputePipeline(cls, *args, **kwargs) -> ComputePipeline:
		return ComputePipeline(*args, **kwargs)


RenderContext = _RenderContext()


class CommandContext:
	"""One independent GPU command encoder.

	Multiple CommandContexts may be encoded independently and later submitted in the
	order in which their resulting command buffers should execute.
	"""

	def __init__(self, label: str | None = None):
		self.handle = RenderContext.device.create_command_encoder(label=label or "")
		self.finished = False

	def render_pass(
		self,
		*,
		color: ColorAttachment | Sequence[ColorAttachment] = (),
		depth: DepthAttachment | None = None,
		label: str | None = None,
	) -> RenderPass:
		return RenderPass(self, color=color, depth=depth, label=label)

	def compute_pass(self, *, label: str | None = None) -> ComputePass:
		return ComputePass(self, label=label)

	def clear_buffer(
		self,
		buffer: GpuBuffer,
		*,
		offset: int = 0,
		size: int | None = None,
	) -> None:
		self.handle.clear_buffer(buffer.handle, offset, size)

	def copy_buffer(
		self,
		source: GpuBuffer,
		destination: GpuBuffer,
		*,
		source_offset: int = 0,
		destination_offset: int = 0,
		size: int | None = None,
	) -> None:
		if size is None:
			size = min(source.handle.size - source_offset, destination.handle.size - destination_offset)
		self.handle.copy_buffer_to_buffer(
			source.handle,
			source_offset,
			destination.handle,
			destination_offset,
			size,
		)

	def finish(self, label: str | None = None) -> wgpu.GPUCommandBuffer:
		if self.finished:
			raise RuntimeError("CommandContext.finish() called more than once")
		self.finished = True
		return self.handle.finish(label=label or "")


class RenderPass:
	def __init__(
		self,
		commands: CommandContext,
		*,
		color: ColorAttachment | Sequence[ColorAttachment] = (),
		depth: DepthAttachment | None = None,
		label: str | None = None,
	):
		self.commands = commands
		self.color = [color] if isinstance(color, ColorAttachment) else list(color)
		self.depth = depth
		self.label = label

		self.handle: wgpu.GPURenderPassEncoder | None = None
		self.pipeline: RenderPipeline | None = None
		self.bind_groups: Dict[int, BindGroup | wgpu.GPUBindGroup] = {}
		self._gpu_pipeline: wgpu.GPURenderPipeline | None = None

	@property
	def color_formats(self) -> tuple[str, ...]:
		return tuple(attachment.format for attachment in self.color)

	@property
	def depth_format(self) -> str | None:
		return self.depth.format if self.depth else None

	def __enter__(self) -> RenderPass:
		self.handle = self.commands.handle.begin_render_pass(
			label=self.label or "",
			color_attachments=[attachment.descriptor() for attachment in self.color],
			depth_stencil_attachment=self.depth.descriptor() if self.depth else None,
		)
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		self.end()

	def end(self) -> None:
		if self.handle is not None:
			self.handle.end()
			self.handle = None

	def set_pipeline(self, pipeline: RenderPipeline) -> None:
		self.pipeline = pipeline
		self._gpu_pipeline = None

	def set_bind_group(
		self,
		index: int,
		bindings: BindGroup | wgpu.GPUBindGroup,
	) -> None:
		self.bind_groups[index] = bindings

	def set_vertex_buffer(
		self,
		slot: int,
		buffer: GpuBuffer,
		*,
		offset: int = 0,
		size: int | None = None,
	) -> None:
		self._require_handle().set_vertex_buffer(slot, buffer.handle, offset, size)

	def set_index_buffer(
		self,
		buffer: GpuBuffer,
		*,
		format: str,
		offset: int = 0,
		size: int | None = None,
	) -> None:
		self._require_handle().set_index_buffer(buffer.handle, format, offset, size)

	def draw(
		self,
		vertex_count: int,
		*,
		instance_count: int = 1,
		first_vertex: int = 0,
		first_instance: int = 0,
	) -> None:
		self._prepare_pipeline(())
		self._require_handle().draw(
			vertex_count,
			instance_count,
			first_vertex,
			first_instance,
		)

	def draw_indexed(
		self,
		index_count: int,
		*,
		instance_count: int = 1,
		first_index: int = 0,
		base_vertex: int = 0,
		first_instance: int = 0,
	) -> None:
		self._prepare_pipeline(())
		self._require_handle().draw_indexed(
			index_count,
			instance_count,
			first_index,
			base_vertex,
			first_instance,
		)

	def draw_mesh(
		self,
		mesh: Mesh,
		*,
		instances: GpuBuffer | None = None,
		instance_count: int | None = None,
	) -> None:
		buffers = [(mesh.vertex_dtype, wgpu.VertexStepMode.vertex)]
		if instances is not None:
			buffers.append((instances.content.dtype, wgpu.VertexStepMode.instance))

		self._prepare_pipeline(buffers)
		handle = self._require_handle()

		vertex_start, vertex_count = mesh.vertex_range
		handle.set_vertex_buffer(
			0,
			mesh.vertex_buffer.handle,
			vertex_start * mesh.vertex_dtype.itemsize,
			vertex_count * mesh.vertex_dtype.itemsize,
		)

		if instances is not None:
			handle.set_vertex_buffer(1, instances.handle)

		index_start, index_count = mesh.index_range
		handle.set_index_buffer(
			mesh.index_buffer.handle,
			mesh.index_format,
			index_start * mesh.index_dtype.itemsize,
			index_count * mesh.index_dtype.itemsize,
		)

		if instance_count is None:
			instance_count = instances.content.size if instances is not None else 1
		handle.draw_indexed(mesh.index_count, instance_count, 0, 0, 0)

	def _prepare_pipeline(
		self,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> None:
		if self.pipeline is None:
			raise RuntimeError("No RenderPipeline has been set on this pass")

		pipeline = self.pipeline.get_pipeline(self, buffers)
		handle = self._require_handle()
		if pipeline is not self._gpu_pipeline:
			handle.set_pipeline(pipeline)
			self._gpu_pipeline = pipeline

		for index, bindings in self.bind_groups.items():
			if isinstance(bindings, BindGroup):
				bindings = bindings.get_handle(pipeline)
			handle.set_bind_group(index, bindings)

	def _require_handle(self) -> wgpu.GPURenderPassEncoder:
		if self.handle is None:
			raise RuntimeError("RenderPass is not active")
		return self.handle


class ComputePass:
	def __init__(self, commands: CommandContext, *, label: str | None = None):
		self.commands = commands
		self.label = label
		self.handle: wgpu.GPUComputePassEncoder | None = None
		self.pipeline: ComputePipeline | None = None

	def __enter__(self) -> ComputePass:
		self.handle = self.commands.handle.begin_compute_pass(label=self.label or "")
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		self.end()

	def end(self) -> None:
		if self.handle is not None:
			self.handle.end()
			self.handle = None

	def set_pipeline(self, pipeline: ComputePipeline) -> None:
		self.pipeline = pipeline
		self._require_handle().set_pipeline(pipeline.handle)

	def set_bind_group(self, index: int, bindings: BindGroup | wgpu.GPUBindGroup) -> None:
		if self.pipeline is None:
			raise RuntimeError("Set the ComputePipeline before its bind groups")
		if isinstance(bindings, BindGroup):
			bindings = bindings.get_handle(self.pipeline.handle)
		self._require_handle().set_bind_group(index, bindings)

	def dispatch(self, x: int, y: int = 1, z: int = 1) -> None:
		self._require_handle().dispatch_workgroups(x, y, z)

	def dispatch_indirect(self, buffer: GpuBuffer, offset: int = 0) -> None:
		self._require_handle().dispatch_workgroups_indirect(buffer.handle, offset)

	def _require_handle(self) -> wgpu.GPUComputePassEncoder:
		if self.handle is None:
			raise RuntimeError("ComputePass is not active")
		return self.handle


# numpy buffers ---------------------------------------------------------------

_VertexFormat: Dict[Any, Tuple[str, int]] = {
	np.int8:	("sint8",   1),
	np.uint8:	("uint8",   1),
	np.int16:	("sint16",  2),
	np.uint16:	("uint16",  2),
	np.int32:	("sint32",  4),
	np.uint32:	("uint32",  4),
	np.int64:	("sint64",  8),
	np.uint64:	("uint64",  8),
	np.uint64:	("uint32",  4),
	np.float16:	("float16", 2),
	np.float32:	("float32", 4),
	np.float64:	("float64", 8),
}


def dtype_to_vertex_format(dtype: np.dtype) -> List | str:
	dtype = np.dtype(dtype)
	if dtype.fields is not None:
		res = []
		for name, (field_dtype, offset) in dtype.fields.items():
			vformat = dtype_to_vertex_format(field_dtype)
			if isinstance(vformat, str):
				res.append((vformat, offset))
			else:
				for vf, local_offset in vformat:
					res.append((vf, offset + local_offset))
		return res

	if dtype.subdtype is None:
		try:
			prefix, scalar_size = _VertexFormat[dtype.type]
		except KeyError:
			raise TypeError(f"Unsupported vertex scalar dtype: {dtype}") from None
		return getattr(wgpu.VertexFormat, prefix)

	base_dtype, shape = dtype.subdtype
	base_type = base_dtype.type
	count = shape[0]
	if count < 1 or count > 4:
		raise ValueError(
			f"VertexFormat arrays must have 1-4 components (found {count} for {dtype.subdtype})"
		)

	try:
		prefix, scalar_size = _VertexFormat[base_type]
	except KeyError:
		raise TypeError(f"Unsupported vertex base dtype: {base_type}") from None

	vformat = getattr(wgpu.VertexFormat, prefix if count == 1 else f"{prefix}x{count}")
	if len(shape) == 1:
		return vformat
	if len(shape) == 2:
		row_count = shape[1]
		return [(vformat, scalar_size * count * i) for i in range(row_count)]

	raise TypeError(
		f"Only 1D or 2D array dtypes are valid vertex attributes (found {shape})"
	)


def _vertex_buffer_layout(
	dtype: np.dtype,
	step_mode: str,
	shader_location: int,
) -> tuple[wgpu.VertexBufferLayout, int]:
	formats = dtype_to_vertex_format(dtype)
	if isinstance(formats, str):
		formats = [(formats, 0)]

	attributes = [
		wgpu.VertexAttribute(
			shader_location=shader_location + i,
			offset=offset,
			format=vertex_format,
		)
		for i, (vertex_format, offset) in enumerate(formats)
	]
	return wgpu.VertexBufferLayout(
		array_stride=np.dtype(dtype).itemsize,
		step_mode=step_mode,
		attributes=attributes,
	), shader_location + len(attributes)


class GpuBuffer:
	# Keeping the numpy reference is convenient for ECS-driven buffers and uploads.
	def __init__(
		self,
		content: np.ndarray,
		usage: wgpu.BufferUsage,
		upload: bool = True,
		label: str | None = None,
	):
		self.content = np.asarray(content)
		self.usage = usage
		self.label = label

		if upload:
			self.handle = RenderContext.device.create_buffer_with_data(
				label=label or "",
				data=self.content,
				usage=self.usage,
			)
		else:
			self.handle = RenderContext.device.create_buffer(
				label=label or "",
				size=max(4, self.content.nbytes),
				usage=self.usage,
			)

	def resize(self, count: int) -> bool:
		size = higher_pow2(max(4, count * self.content.itemsize))
		if size <= self.handle.size:
			return False

		self.handle = RenderContext.device.create_buffer(
			label=self.label or "",
			size=size,
			usage=self.usage,
		)
		return True

	def upload(self) -> None:
		RenderContext.device.queue.write_buffer(self.handle, 0, self.content)

	def upload_range(self, start: int, count: int) -> None:
		if count <= 0:
			return
		uploaded = self.content[start:start + count]
		RenderContext.device.queue.write_buffer(
			buffer=self.handle,
			buffer_offset=start * self.content.itemsize,
			data=uploaded,
		)

	def write(self, data: np.ndarray, *, offset: int = 0) -> None:
		self.content = np.asarray(data)
		RenderContext.device.queue.write_buffer(self.handle, offset, self.content)


class GpuBufferPool:
	def __init__(
		self,
		dtype: np.dtype,
		usage: wgpu.BufferUsage,
		initial_count: int = 0,
	):
		self.dtype = np.dtype(dtype)
		self.used = 0
		self.buffer = GpuBuffer(
			np.empty(initial_count, dtype=self.dtype),
			usage | wgpu.BufferUsage.COPY_DST,
			upload=False,
		)

	def alloc(self, data: np.ndarray) -> Tuple[int, int]:
		start = self.used
		count = data.size
		end = start + count

		self._ensure_capacity(end)
		self.buffer.content[start:end] = data
		self.buffer.upload_range(start, count)
		self.used = end
		return start, count

	def _ensure_capacity(self, count: int) -> None:
		old_content = self.buffer.content
		if count <= old_content.size:
			return

		new_count = higher_pow2(max(1, count))
		new_content = np.empty(new_count, dtype=self.dtype)
		new_content[:self.used] = old_content[:self.used]
		self.buffer.content = new_content

		if self.buffer.resize(new_count):
			self.buffer.upload_range(0, self.used)


class Mesh:
	# Cache geometry with the same dtype in shared pools, as in the original implementation.
	vertex_buffers: Dict[np.dtype, GpuBufferPool] = {}
	index_buffers: Dict[np.dtype, GpuBufferPool] = {}

	index_formats = {
		np.dtype(np.uint16): wgpu.IndexFormat.uint16,
		np.dtype(np.uint32): wgpu.IndexFormat.uint32,
	}

	def __init__(self, vertex_data: np.ndarray, indices: np.ndarray):
		self.index_count = indices.size
		self.vertex_dtype = np.dtype(vertex_data.dtype)
		self.index_dtype = np.dtype(indices.dtype)

		if self.index_dtype not in self.index_formats:
			raise TypeError(f"Unsupported index dtype: {self.index_dtype}")

		vertex_pool = self.vertex_buffers.get(self.vertex_dtype)
		if vertex_pool is None:
			vertex_pool = GpuBufferPool(self.vertex_dtype, wgpu.BufferUsage.VERTEX)
			self.vertex_buffers[self.vertex_dtype] = vertex_pool

		index_pool = self.index_buffers.get(self.index_dtype)
		if index_pool is None:
			index_pool = GpuBufferPool(self.index_dtype, wgpu.BufferUsage.INDEX)
			self.index_buffers[self.index_dtype] = index_pool

		self.vertex_buffer = vertex_pool.buffer
		self.vertex_range = vertex_pool.alloc(vertex_data)

		self.index_buffer = index_pool.buffer
		self.index_range = index_pool.alloc(indices)
		self.index_format = self.index_formats[self.index_dtype]


# textures --------------------------------------------------------------------

class Texture:
	def __init__(
		self,
		size: tuple[int, int] | tuple[int, int, int],
		*,
		format: str,
		usage: wgpu.TextureUsage,
		mip_level_count: int = 1,
		label: str | None = None,
	):
		if len(size) == 2:
			size = (*size, 1)
		self.size = size
		self.format = format
		self.usage = usage
		self.handle = RenderContext.device.create_texture(
			label=label or "",
			size=size,
			format=format,
			usage=usage,
			mip_level_count=mip_level_count,
		)

	def view(self, **kwargs) -> TextureView:
		return TextureView(self, self.handle.create_view(**kwargs))

	def color_attachment(
		self,
		*,
		clear: Color | None = None,
		store: bool = True,
	) -> ColorAttachment:
		return ColorAttachment(self.handle.create_view(), self.format, clear, store)

	def depth_attachment(
		self,
		*,
		clear: float | None = None,
		store: bool = True,
		read_only: bool = False,
	) -> DepthAttachment:
		return DepthAttachment(self.handle.create_view(), self.format, clear, store, read_only)


@dataclass(frozen=True)
class TextureView:
	texture: Texture
	handle: wgpu.GPUTextureView

	@property
	def format(self) -> str:
		return self.texture.format


class Sampler:
	def __init__(self, **descriptor: Any):
		self.handle = RenderContext.device.create_sampler(**descriptor)


# WGSL shader metadata --------------------------------------------------------

@dataclass(frozen=True)
class ShaderAttribute:
	name: str
	value: str | None = None

	def int_value(self) -> int:
		if self.value is None:
			raise ValueError(f"Shader attribute @{self.name} has no value")
		return int(self.value)


@dataclass(frozen=True)
class ShaderField:
	name: str
	type_name: str
	attributes: tuple[ShaderAttribute, ...] = ()

	def attribute(self, name: str) -> ShaderAttribute | None:
		return next(
			(attribute for attribute in self.attributes if attribute.name == name),
			None,
		)


@dataclass(frozen=True)
class ShaderStruct:
	name: str
	fields: tuple[ShaderField, ...]


@dataclass(frozen=True)
class BindingInfo:
	name: str
	group: int
	binding: int
	storage: str | None
	access: str | None
	type_name: str


@dataclass(frozen=True)
class EntryPointInfo:
	name: str
	stage: str
	parameters: tuple[ShaderField, ...]
	return_type: str | None
	return_attributes: tuple[ShaderAttribute, ...]


@dataclass(frozen=True)
class UniformInfo:
	name: str
	group: int
	binding: int
	struct_name: str
	dtype: np.dtype


@dataclass(frozen=True)
class VertexInfo:
	name: str
	location: int
	type_name: str
	dtype: np.dtype


@dataclass(frozen=True)
class ShaderInterface:
	structs: Dict[str, ShaderStruct]
	bindings: Dict[str, BindingInfo]
	entry_points: Dict[str, EntryPointInfo]


@dataclass(frozen=True)
class _WGSLType:
	dtype: np.dtype
	align: int
	size: int


_SCALAR_TYPES = {
	"f32": (np.dtype(np.float32), 4, 4),
	"i32": (np.dtype(np.int32), 4, 4),
	"u32": (np.dtype(np.uint32), 4, 4),
}


def _round_up(alignment: int, value: int) -> int:
	return (value + alignment - 1) // alignment * alignment


def _strip_wgsl_comments(source: str) -> str:
	source = re.sub(r"/\*[\s\S]*?\*/", "", source)
	return re.sub(r"//.*", "", source)


class ShaderSource:
	_ATTRIBUTE = r"@\w+(?:\s*\([^)]*\))?"
	_ATTRIBUTES = rf"(?:{_ATTRIBUTE}\s*)*"
	_PARENS = r"(?:[^()]|\([^()]*\))*"

	_attribute_pattern = re.compile(
		r"@(?P<name>\w+)(?:\s*\(\s*(?P<value>[^)]*?)\s*\))?"
	)

	_struct_pattern = re.compile(
		r"\bstruct\s+(?P<name>\w+)\s*\{(?P<body>.*?)\}\s*;?",
		re.S,
	)

	_field_pattern = re.compile(
		rf"(?P<attributes>{_ATTRIBUTES})(?P<name>\w+)\s*:\s*"
		rf"(?P<type>.+?)"
		rf"(?=,\s*(?:{_ATTRIBUTE}\s*)*\w+\s*:|,\s*$|\s*$)",
		re.S,
	)

	_binding_pattern = re.compile(
		rf"(?P<attributes>{_ATTRIBUTES})"
		r"var(?:\s*<\s*(?P<storage>[^>]+?)\s*>)?\s+"
		r"(?P<name>\w+)\s*:\s*(?P<type>[^;]+?)\s*;",
		re.S,
	)

	_entry_pattern = re.compile(
		rf"(?P<attributes>{_ATTRIBUTES})fn\s+(?P<name>\w+)\s*"
		rf"\((?P<parameters>{_PARENS})\)\s*"
		rf"(?:->\s*(?P<return_attributes>{_ATTRIBUTES})"
		r"(?P<return_type>[^{]+?))?\s*\{",
		re.S,
	)

	_parameter_pattern = re.compile(
		rf"(?P<attributes>{_ATTRIBUTES})(?P<name>\w+)\s*:\s*"
		rf"(?P<type>.+?)"
		rf"(?=,\s*(?:{_ATTRIBUTE}\s*)*\w+\s*:|\s*$)",
		re.S,
	)

	_storage_pattern = re.compile(
		r"^\s*(?P<storage>\w+)(?:\s*,\s*(?P<access>\w+))?\s*$"
	)

	def __init__(self, source: str):
		self.source = source
		self.interface = self._parse(_strip_wgsl_comments(source))
		self.uniforms = self._uniforms()

	@property
	def structs(self) -> Dict[str, ShaderStruct]:
		return self.interface.structs

	@property
	def bindings(self) -> Dict[str, BindingInfo]:
		return self.interface.bindings

	@property
	def entry_points(self) -> Dict[str, tuple[str, ...]]:
		entries = {"vertex": [], "fragment": [], "compute": []}

		for entry in self.interface.entry_points.values():
			entries[entry.stage].append(entry.name)

		return {
			stage: tuple(names)
			for stage, names in entries.items()
		}

	@property
	def vertex_dtype(self) -> np.dtype | None:
		entries = self.entry_points["vertex"]
		if not entries:
			return None

		dtype = self.vertex_dtype_for(entries[0])

		for entry in entries[1:]:
			if self.vertex_dtype_for(entry) != dtype:
				raise ValueError(
					"Shader vertex entry points use different input layouts; "
					"use vertex_dtype_for(entry_point)"
				)

		return dtype

	def vertex_dtype_for(self, entry_point: str) -> np.dtype:
		return np.dtype([
			(field.name, field.dtype)
			for field in self.vertex_inputs(entry_point)
		])

	def vertex_inputs(self, entry_point: str) -> tuple[VertexInfo, ...]:
		try:
			entry = self.interface.entry_points[entry_point]
		except KeyError:
			raise ValueError(f"Shader has no entry point {entry_point!r}") from None

		if entry.stage != "vertex":
			raise ValueError(
				f"Shader entry point {entry_point!r} is not a vertex shader"
			)

		fields = []

		for parameter in entry.parameters:
			location = parameter.attribute("location")

			if location is not None:
				fields.append(self._vertex_info(parameter, location))
				continue

			struct = self.structs.get(parameter.type_name)
			if struct is None:
				continue

			for field in struct.fields:
				location = field.attribute("location")
				if location is not None:
					fields.append(self._vertex_info(field, location))

		fields.sort(key=lambda field: field.location)

		locations = [field.location for field in fields]
		if len(locations) != len(set(locations)):
			raise ValueError(
				f"Vertex entry point {entry_point!r} contains duplicate @location values"
			)

		return tuple(fields)

	def uniform_dtype(self, struct_name: str) -> np.dtype:
		return self._uniform_struct_dtype(struct_name)

	def _parse(self, source: str) -> ShaderInterface:
		structs = {}

		for match in self._struct_pattern.finditer(source):
			fields = tuple(
				self._field(field)
				for field in self._field_pattern.finditer(match.group("body"))
			)

			structs[match.group("name")] = ShaderStruct(
				match.group("name"),
				fields,
			)

		bindings = {}

		for match in self._binding_pattern.finditer(source):
			attributes = self._attributes(match.group("attributes"))
			group = self._attribute(attributes, "group")
			binding = self._attribute(attributes, "binding")

			if group is None or binding is None:
				continue

			storage = None
			access = None
			storage_text = match.group("storage")

			if storage_text is not None:
				storage_match = self._storage_pattern.fullmatch(storage_text)

				if storage_match is None:
					raise ValueError(
						f"Cannot parse WGSL address space {storage_text!r}"
					)

				storage = storage_match.group("storage")
				access = storage_match.group("access")

			name = match.group("name")

			bindings[name] = BindingInfo(
				name=name,
				group=group.int_value(),
				binding=binding.int_value(),
				storage=storage,
				access=access,
				type_name=self._type_name(match.group("type")),
			)

		entry_points = {}

		for match in self._entry_pattern.finditer(source):
			attributes = self._attributes(match.group("attributes"))

			stage = next(
				(
					stage
					for stage in ("vertex", "fragment", "compute")
					if self._attribute(attributes, stage) is not None
				),
				None,
			)

			if stage is None:
				continue

			parameters = tuple(
				self._field(parameter)
				for parameter in self._parameter_pattern.finditer(
					match.group("parameters")
				)
			)

			name = match.group("name")
			return_type = match.group("return_type")

			entry_points[name] = EntryPointInfo(
				name=name,
				stage=stage,
				parameters=parameters,
				return_type=self._type_name(return_type) if return_type else None,
				return_attributes=self._attributes(
					match.group("return_attributes") or ""
				),
			)

		return ShaderInterface(
			structs=structs,
			bindings=bindings,
			entry_points=entry_points,
		)

	def _field(self, match: re.Match) -> ShaderField:
		return ShaderField(
			name=match.group("name"),
			type_name=self._type_name(match.group("type")),
			attributes=self._attributes(match.group("attributes")),
		)

	def _attributes(self, text: str) -> tuple[ShaderAttribute, ...]:
		return tuple(
			ShaderAttribute(
				match.group("name"),
				match.group("value"),
			)
			for match in self._attribute_pattern.finditer(text)
		)

	@staticmethod
	def _attribute(
		attributes: tuple[ShaderAttribute, ...],
		name: str,
	) -> ShaderAttribute | None:
		return next(
			(attribute for attribute in attributes if attribute.name == name),
			None,
		)

	@staticmethod
	def _type_name(type_name: str) -> str:
		return re.sub(r"\s+", "", type_name)

	def _uniforms(self) -> Dict[str, UniformInfo]:
		uniforms = {}

		for binding in self.bindings.values():
			if binding.storage != "uniform":
				continue

			if binding.type_name not in self.structs:
				raise TypeError(
					f"Uniform {binding.name!r} must refer to a named WGSL struct"
				)

			uniforms[binding.name] = UniformInfo(
				name=binding.name,
				group=binding.group,
				binding=binding.binding,
				struct_name=binding.type_name,
				dtype=self._uniform_struct_dtype(binding.type_name),
			)

		return uniforms

	def _uniform_struct_dtype(
		self,
		name: str,
		stack: tuple[str, ...] = (),
	) -> np.dtype:
		if name in stack:
			raise TypeError(
				f"Recursive WGSL struct {name!r} cannot be a uniform dtype"
			)

		try:
			struct = self.structs[name]
		except KeyError:
			raise TypeError(f"Unknown WGSL struct {name!r}") from None

		names = []
		formats = []
		offsets = []

		offset = 0
		struct_align = 16

		for field in struct.fields:
			field_type = self._uniform_type(
				field.type_name,
				stack + (name,),
			)

			align = (
				max(16, field_type.align)
				if field.type_name in self.structs
				else field_type.align
			)

			offset = _round_up(align, offset)

			names.append(field.name)
			formats.append(field_type.dtype)
			offsets.append(offset)

			offset += field_type.size
			struct_align = max(struct_align, align)

		return np.dtype({
			"names": names,
			"formats": formats,
			"offsets": offsets,
			"itemsize": _round_up(struct_align, offset),
		})

	def _uniform_type(
		self,
		type_name: str,
		stack: tuple[str, ...],
	) -> _WGSLType:
		if type_name in _SCALAR_TYPES:
			dtype, align, size = _SCALAR_TYPES[type_name]
			return _WGSLType(dtype, align, size)

		vector_alias = re.fullmatch(r"vec([234])([fiu])", type_name)

		if vector_alias:
			scalar = {
				"f": "f32",
				"i": "i32",
				"u": "u32",
			}[vector_alias.group(2)]

			type_name = f"vec{vector_alias.group(1)}<{scalar}>"

		matrix_alias = re.fullmatch(r"mat([234])x([234])f", type_name)

		if matrix_alias:
			type_name = (
				f"mat{matrix_alias.group(1)}x"
				f"{matrix_alias.group(2)}<f32>"
			)

		vector = re.fullmatch(r"vec([234])<([fiu]32)>", type_name)

		if vector:
			count = int(vector.group(1))
			base, _, scalar_size = _SCALAR_TYPES[vector.group(2)]
			align = 8 if count == 2 else 16

			return _WGSLType(
				np.dtype((base, (count,))),
				align,
				scalar_size * count,
			)

		matrix = re.fullmatch(r"mat([234])x([234])<f32>", type_name)

		if matrix:
			columns, rows = map(int, matrix.groups())
			column = self._uniform_type(f"vec{rows}<f32>", stack)
			stride = _round_up(column.align, column.size)

			if stride != column.size:
				raise TypeError(
					f"{type_name} requires padding inside matrix columns; "
					"this uniform layout is not implemented yet"
				)

			return _WGSLType(
				np.dtype((np.float32, (columns, rows))),
				column.align,
				stride * columns,
			)

		array = re.fullmatch(r"array<(.+),(\d+)>", type_name)

		if array:
			member = self._uniform_type(array.group(1), stack)
			count = int(array.group(2))

			align = max(16, member.align)
			stride = _round_up(align, member.size)

			if stride != member.dtype.itemsize:
				raise TypeError(
					f"{type_name} requires per-element padding; "
					"this uniform layout is not implemented yet"
				)

			return _WGSLType(
				np.dtype((member.dtype, (count,))),
				align,
				stride * count,
			)

		if type_name in self.structs:
			dtype = self._uniform_struct_dtype(type_name, stack)
			return _WGSLType(dtype, 16, dtype.itemsize)

		raise TypeError(f"Unsupported WGSL uniform type {type_name!r}")

	def _vertex_info(
		self,
		field: ShaderField,
		location: ShaderAttribute,
	) -> VertexInfo:
		return VertexInfo(
			name=field.name,
			location=location.int_value(),
			type_name=field.type_name,
			dtype=self._vertex_type_dtype(field.type_name),
		)

	def _vertex_type_dtype(self, type_name: str) -> np.dtype:
		if type_name in _SCALAR_TYPES:
			return _SCALAR_TYPES[type_name][0]

		alias = re.fullmatch(r"vec([234])([fiu])", type_name)

		if alias:
			scalar = {
				"f": "f32",
				"i": "i32",
				"u": "u32",
			}[alias.group(2)]

			type_name = f"vec{alias.group(1)}<{scalar}>"

		vector = re.fullmatch(r"vec([234])<([fiu]32)>", type_name)

		if vector:
			count = int(vector.group(1))
			base = _SCALAR_TYPES[vector.group(2)][0]

			return np.dtype((base, (count,)))

		raise TypeError(
			f"Unsupported WGSL vertex type {type_name!r}"
		)


class Shader:

	def __init__(
		self,
		*,
		source: str | None = None,
		filepath: str | None = None,
		basedir: str | None = None,
		features: Sequence[str] | None = None,
		params: Dict[str, Any] | None = None,
		label: str | None = None,
	):
		if source is None and filepath is None:
			raise ValueError("Shader requires source= or filepath=")

		self.source = self._load_source(
			source,
			filepath,
			basedir,
			features,
			params,
		)

		self.info = ShaderSource(self.source)

		self.module = RenderContext.device.create_shader_module(
			label=label or filepath or "shader",
			code=self.source,
		)

		self.structs = self.info.structs
		self.bindings = self.info.bindings
		self.uniforms = self.info.uniforms
		self.entry_points = self.info.entry_points

	@property
	def vertex_dtype(self) -> np.dtype | None:
		return self.info.vertex_dtype

	def vertex_dtype_for(self, entry_point: str) -> np.dtype:
		return self.info.vertex_dtype_for(entry_point)

	def vertex_inputs(self, entry_point: str) -> tuple[VertexInfo, ...]:
		return self.info.vertex_inputs(entry_point)

	def UniformBuffer(self, name: str | None = None) -> UniformBuffer:
		if name is None:
			if len(self.uniforms) != 1:
				raise ValueError(
					"Uniform name is required when a shader has "
					"multiple uniform blocks"
				)

			name = next(iter(self.uniforms))

		return UniformBuffer(self.uniforms[name])

	def entry(
		self,
		stage: str,
		name: str | None = None,
	) -> str | None:
		entries = self.entry_points.get(stage, ())

		if name is not None:
			if name not in entries:
				raise ValueError(
					f"Shader has no {stage} entry point {name!r}"
				)

			return name

		if not entries:
			return None

		if len(entries) > 1:
			raise ValueError(
				f"Shader has multiple {stage} entry points; "
				"specify one explicitly"
			)

		return entries[0]

	def _load_source(
		self,
		source: str | None,
		filepath: str | None,
		basedir: str | None,
		features: Sequence[str] | None,
		params: Dict[str, Any] | None,
	) -> str:
		text = (
			source
			if source is not None
			else open(filepath, "r").read()
		)

		base = (
			basedir
			if basedir
			else os.path.dirname(os.path.abspath(filepath))
			if filepath
			else glob("**/shaders/", recursive=True)
		)

		env = Environment(
			loader=FileSystemLoader(base),
			undefined=StrictUndefined,
			autoescape=False,
			keep_trailing_newline=True,
			trim_blocks=True,
			lstrip_blocks=True,
			line_statement_prefix="#",
		)

		template = env.from_string(text)
		feature_dict = DefaultFalseDict()

		if features is not None:
			feature_dict.update({
				feature: True
				for feature in features
			})

		return template.render(
			FEATURES=feature_dict,
			PARAMS=params or {},
		)


class DefaultFalseDict(Dict):
	def __missing__(self, key):
		return False


class UniformBuffer(GpuBuffer):
	def __init__(self, info: UniformInfo):
		self.info = info

		super().__init__(
			np.zeros(1, dtype=info.dtype),
			wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
		)


# pipelines / bindings --------------------------------------------------------

class BindGroup:
	def __init__(
		self,
		shader: Shader,
		group: int,
		resources: Dict[str, Any],
	):
		self.shader = shader
		self.group = group
		self.resources = resources
		self._cache: Dict[int, wgpu.GPUBindGroup] = {}

		for name in resources:
			binding = shader.bindings.get(name)
			if binding is None:
				raise KeyError(f"Shader has no binding named {name!r}")
			if binding.group != group:
				raise ValueError(
					f"Binding {name!r} belongs to group {binding.group}, not group {group}"
				)

	def get_handle(
		self,
		pipeline: wgpu.GPURenderPipeline | wgpu.GPUComputePipeline,
	) -> wgpu.GPUBindGroup:
		key = id(pipeline)
		cached = self._cache.get(key)
		if cached is not None:
			return cached

		layout = pipeline.get_bind_group_layout(self.group)

		entries = []
		for name, resource in self.resources.items():
			binding = self.shader.bindings[name]
			entries.append(wgpu.BindGroupEntry(
				binding=binding.binding,
				resource=self._resource(resource),
			))

		handle = RenderContext.device.create_bind_group(
			layout=layout,
			entries=entries,
		)
		self._cache[key] = handle
		return handle

	def _resource(self, resource: Any) -> Any:
		if isinstance(resource, GpuBuffer):
			return wgpu.BufferBinding(
				buffer=resource.handle,
				offset=0,
				size=resource.content.nbytes,
			)
		if isinstance(resource, Texture):
			return resource.handle.create_view()
		if isinstance(resource, TextureView):
			return resource.handle
		if isinstance(resource, Sampler):
			return resource.handle
		return resource


class RenderPipeline:
	"""Render pipeline specification.

	A concrete GPURenderPipeline is cached for each combination of inferred vertex
	layouts and render-target formats. The user never has to specify those layouts.
	"""

	def __init__(
		self,
		shader: Shader,
		*,
		vertex_entry: str | None = None,
		fragment_entry: str | None = None,
		topology: str = wgpu.PrimitiveTopology.triangle_list,
		cull_mode: str = wgpu.CullMode.back,
		front_face: str = wgpu.FrontFace.ccw,
		depth_test: str | None = wgpu.CompareFunction.less,
		depth_write: bool = True,
		depth_bias: int = 0,
		depth_bias_slope_scale: float = 0.0,
		blend: wgpu.BlendState | None = None,
		label: str | None = None,
	):
		self.shader = shader
		self.vertex_entry = shader.entry("vertex", vertex_entry)
		self.fragment_entry = shader.entry("fragment", fragment_entry)
		self.topology = topology
		self.cull_mode = cull_mode
		self.front_face = front_face
		self.depth_test = depth_test
		self.depth_write = depth_write
		self.depth_bias = depth_bias
		self.depth_bias_slope_scale = depth_bias_slope_scale
		self.blend = blend
		self.label = label

		if self.vertex_entry is None:
			raise ValueError("RenderPipeline shader has no @vertex entry point")

		vertex_inputs = shader.vertex_inputs(self.vertex_entry)

		self.vertex_inputs = {
			field.name: field
			for field in vertex_inputs
		}

		if len(self.vertex_inputs) != len(vertex_inputs):
			raise ValueError(
				f"Vertex entry point {self.vertex_entry!r} contains "
				"duplicate input field names"
			)

		self._cache: Dict[tuple, wgpu.GPURenderPipeline] = {}
		self._cache_lock = Lock()

	def bind_group(self, group: int, **resources: Any) -> BindGroup:
		return BindGroup(self.shader, group, resources)

	def get_pipeline(
		self,
		render_pass: RenderPass,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> wgpu.GPURenderPipeline:
		key = (
			tuple((repr(np.dtype(dtype).descr), step_mode) for dtype, step_mode in buffers),
			render_pass.color_formats,
			render_pass.depth_format,
		)
		cached = self._cache.get(key)
		if cached is not None:
			return cached

		with self._cache_lock:
			cached = self._cache.get(key)
			if cached is None:
				cached = self._make_pipeline(render_pass, buffers)
				self._cache[key] = cached
		return cached

	def _make_pipeline(
		self,
		render_pass: RenderPass,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> wgpu.GPURenderPipeline:
		provided_vertex_input = set()
		layouts = []

		for dtype, step_mode in buffers:
			if dtype.fields is None:
				raise TypeError(f"Vertex buffer dtype must be structured, got {dtype}")

			attributes = []

			for name, (field_dtype, offset) in dtype.fields.items():
				shader_input = self.vertex_inputs.get(name)

				if shader_input is None:
					continue

				if name in provided_vertex_input:
					raise ValueError(
						f"Vertex attribute {name!r} is provided by multiple buffers"
					)

				field_dtype = np.dtype(field_dtype)
				shader_dtype = shader_input.dtype


				field_size = int(np.prod(field_dtype.shape)) if field_dtype.shape else 1
				shader_size = int(np.prod(shader_dtype.shape)) if shader_dtype.shape else 1

				if (
					field_size != shader_size
					or field_dtype.base.kind != shader_dtype.base.kind
				):
					raise TypeError(
						f"Vertex attribute {name!r} has dtype {field_dtype}; "
						f"shader expects {shader_input.type_name}"
					)

				attributes.append(wgpu.VertexAttribute(
					shader_location=shader_input.location,
					offset=offset,
					format=dtype_to_vertex_format(field_dtype),
				))
				provided_vertex_input.add(name)

			layouts.append(wgpu.VertexBufferLayout(
				array_stride=dtype.itemsize,
				step_mode=step_mode,
				attributes=attributes,
			))

		missing = self.vertex_inputs.keys() - provided_vertex_input
		if missing:
			raise ValueError(
				f"Vertex shader {self.vertex_entry!r} requires attributes "
				f"not provided by the draw buffers: {', '.join(sorted(missing))}"
			)

		fragment = None

		if render_pass.color_formats:
			if self.fragment_entry is None:
				raise ValueError(
					"Rendering to a color target requires an @fragment entry point"
				)

			fragment = wgpu.FragmentState(
				module=self.shader.module,
				entry_point=self.fragment_entry,
				targets=[
					wgpu.ColorTargetState(
						format=format,
						blend=self.blend,
					)
					for format in render_pass.color_formats
				],
			)

		depth_stencil = None

		if self.depth_test is not None:
			if render_pass.depth_format is None:
				raise ValueError(
					"Pipeline has depth testing enabled but pass has no depth target"
				)

			depth_stencil = wgpu.DepthStencilState(
				format=render_pass.depth_format,
				depth_write_enabled=self.depth_write,
				depth_compare=self.depth_test,
				depth_bias=self.depth_bias,
				depth_bias_slope_scale=self.depth_bias_slope_scale,
			)

		return RenderContext.device.create_render_pipeline(
			label=self.label or "",
			layout="auto",
			vertex=wgpu.VertexState(
				module=self.shader.module,
				entry_point=self.vertex_entry,
				buffers=layouts,
			),
			primitive=wgpu.PrimitiveState(
				topology=self.topology,
				cull_mode=self.cull_mode,
				front_face=self.front_face,
			),
			depth_stencil=depth_stencil,
			fragment=fragment,
		)


class ComputePipeline:
	def __init__(
		self,
		shader: Shader,
		*,
		entry: str | None = None,
		label: str | None = None,
	):
		self.shader = shader
		self.entry = shader.entry("compute", entry)
		if self.entry is None:
			raise ValueError("ComputePipeline shader has no @compute entry point")

		self.handle = RenderContext.device.create_compute_pipeline(
			label=label or "",
			layout="auto",
			compute=wgpu.ProgrammableStage(
				module=shader.module,
				entry_point=self.entry,
			),
		)

	def bind_group(self, group: int, **resources: Any) -> BindGroup:
		return BindGroup(self.shader, group, resources)


# compatibility-ish helpers ---------------------------------------------------

def create_depth_framebuffer(
	width: int,
	height: int,
	format: str = wgpu.TextureFormat.depth24plus,
) -> Texture:
	return Texture(
		(width, height),
		format=format,
		usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
	)


def create_depth_sampler() -> Sampler:
	return Sampler(
		address_mode_u=wgpu.AddressMode.clamp_to_edge,
		address_mode_v=wgpu.AddressMode.clamp_to_edge,
		address_mode_w=wgpu.AddressMode.clamp_to_edge,
		compare=wgpu.CompareFunction.less_equal,
		min_filter=wgpu.FilterMode.linear,
		mag_filter=wgpu.FilterMode.linear,
	)


def build_shader_program(shaderPath: str, **kwargs) -> Shader:
	return RenderContext.Shader(filepath=shaderPath, **kwargs)
