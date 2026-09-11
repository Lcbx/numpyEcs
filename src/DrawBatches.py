from dataclasses import dataclass

import numpy as np
import wgpu

from RenderContext import GpuBuffer, Texture, Shader, ComputePipeline
from Utils import mesh_instance_dtype as instance_dtype


uniform_dtype = np.dtype([("view", "<f4", (4, 4)), ("proj", "<f4", (4, 4)), ("light_dir", "<f4", 4)])
mesh_metadata_dtype = np.dtype([("box_center", "<f4", 4), ("box_extents", "<f4", 4)])
batch_dtype = np.dtype([("mesh_id", "<u4"), ("instance_offset", "<u4"), ("instance_count", "<u4"), ("padding", "<u4")])
workgroup_dtype = np.dtype([("batch_id", "<u4"), ("instance_offset", "<u4")])
indirect_dtype = np.dtype([
	("index_count", "<u4"), ("instance_count", "<u4"), ("first_index", "<u4"),
	("base_vertex", "<i4"), ("first_instance", "<u4"),
])


@dataclass
class DrawBatch:
	mesh_id: int
	shader_id: int
	offset: int
	count: int
	command_index: int


@dataclass
class ShaderPass:
	"""Groups 0/1 follow draw_common.shaderlib; extra bindings use groups >= 2.

	Prepass and main must agree on positions, coverage, and rasterization.
	Additional instance attributes can use storage in an extra group, indexed
	by visible_instances[instance_idx] in the current DrawBatches.entities order.
	"""
	pipeline: object
	bindings: tuple = ()


@dataclass
class RenderShader:
	prepass: ShaderPass
	main: ShaderPass


@dataclass
class MeshInfo:
	mesh: object
	box_center: np.ndarray
	box_extents: np.ndarray


class DrawBatches:
	"""Persistent draw resources; record passes explicitly using the caller's commands."""

	def __init__(self, cull_shader : Shader|None = None):
		self._cull_shader = cull_shader or Shader(filepath='scenes/shaders/cull.shader', label="cull")
		self._cull_pipelines = {}
		self.meshes = {}
		self.mesh_rows = {}
		self._mesh_commands = {}
		self._dirty_meshes = set()
		self.shaders = {}
		self.buffers = {}
		self.draw_batches = []
		self.entities = np.empty(0, dtype=np.uint64)
		self._batch_version = None
		self.instance_order_version = 0
		self.bindings = {}
		self.workgroup_count = 0
		storage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
		for name, dtype in (
			("instances", instance_dtype), ("frustum_visible_instances", np.dtype("<u4")),
			("main_visible_instances", np.dtype("<u4")), ("mesh_metadata", mesh_metadata_dtype),
			("batches", batch_dtype), ("workgroups", workgroup_dtype),
			("prepass_draw_cmd", indirect_dtype), ("main_draw_cmd", indirect_dtype),
		):
			usage = storage | (wgpu.BufferUsage.INDIRECT if dtype == indirect_dtype else 0)
			self.buffers[name] = GpuBuffer(np.zeros(1, dtype=dtype), usage, upload=False, label=name)
		usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
		self.prepass_uniform_buffer = GpuBuffer(np.zeros(1, dtype=uniform_dtype), usage)
		self.uniform_buffer = GpuBuffer(np.zeros(1, dtype=uniform_dtype), usage)
		self.camera_params_buffer = self._cull_shader.UniformBuffer("camera_params")

	def get_cull_pipeline(self, pipeline_name : str):
		if (pipeline := self._cull_pipelines.get(pipeline_name)) is None:
			pipeline = self._cull_pipelines[pipeline_name] = ComputePipeline(self._cull_shader, entry=pipeline_name, label=pipeline_name)
		return pipeline

	def _reserve_buffer(self, name, count):
		"""Keep CPU capacity stable between growths; GpuBuffer owns GPU growth.

		Callers replace active inputs after growth. GPU outputs are regenerated.
		"""
		buffer = self.buffers[name]
		if count > buffer.content.size:
			capacity = 1 << (count - 1).bit_length()
			buffer.content = np.zeros(capacity, dtype=buffer.content.dtype)
			buffer.resize(capacity)
		return buffer.content[:count]

	def _upload_array(self, name, values):
		self._reserve_buffer(name, len(values))[:] = values
		if len(values): self.buffers[name].upload_range(0, len(values))

	def register_mesh(self, mesh_id, mesh, bounds=None):
		"""Use explicit conservative local bounds for displaced geometry.

		Re-register after changing mesh geometry, bounds, or pooled draw ranges.
		Only changed metadata/commands are uploaded during sync_batches.
		Registration does not invalidate entity grouping.
		Registry IDs need not be dense.
		"""
		if bounds is None:
			pos = mesh.vertices["position"]
			box_min, box_max = np.min(pos, axis=0), np.max(pos, axis=0)
			bounds = (box_min + box_max) * 0.5, (box_max - box_min) * 0.5
		center, extents = (np.asarray(value, dtype=np.float32).copy() for value in bounds)
		if center.shape != (3,) or extents.shape != (3,) or not np.all(np.isfinite([center, extents])) or np.any(extents < 0):
			raise ValueError("Mesh bounds must be finite vec3 center and nonnegative extents")
		self.meshes[mesh_id] = MeshInfo(mesh, center, extents)
		self._dirty_meshes.add(mesh_id)

	def register_shader(self, shader_id, shader):
		"""Replace pipelines/bindings without invalidating entity grouping."""
		for spec in (shader.prepass, shader.main):
			if any(group < 2 for group, _ in spec.bindings):
				raise ValueError("Groups 0 and 1 are reserved for frame and instance data")
		self.shaders[shader_id] = shader
		for name, spec, uniform, visible in (
			("prepass", shader.prepass, self.prepass_uniform_buffer, "frustum_visible_instances"),
			("main", shader.main, self.uniform_buffer, "main_visible_instances"),
		):
			self.bindings[shader_id, name] = (
				spec.pipeline.shader.bind_group(0, uniforms=uniform),
				spec.pipeline.shader.bind_group(1, instances=self.buffers["instances"], visible_instances=self.buffers[visible]),
			)

	def _sync_meshes(self):
		"""Refresh registered meshes before recording reset/culling commands."""
		for mesh_id in self._dirty_meshes:
			if mesh_id not in self.mesh_rows: continue
			info = self.meshes[mesh_id]
			row = self.mesh_rows[mesh_id]
			buffer = self.buffers["mesh_metadata"]
			metadata = buffer.content[row]
			if not (np.array_equal(metadata["box_center"][:3], info.box_center) and np.array_equal(metadata["box_extents"][:3], info.box_extents)):
				metadata["box_center"][:3], metadata["box_extents"][:3] = info.box_center, info.box_extents
				buffer.upload_range(row, 1)
			mesh = info.mesh
			fields = ("index_count", "first_index", "base_vertex")
			values = (mesh.index_count, mesh.index_range[0], mesh.vertex_range[0])
			for command_index in self._mesh_commands[mesh_id]:
				for name in ("prepass_draw_cmd", "main_draw_cmd"):
					buffer = self.buffers[name]
					command = buffer.content[command_index]
					if all(command[field] == value for field, value in zip(fields, values)): continue
					for field, value in zip(fields, values): command[field] = value
					buffer.upload_range(command_index, 1)
		self._dirty_meshes.clear()

	def sync_batches(self, world, transform_type, mesh_ref_type):
		"""Rebuild on membership or mesh/shader writes; return whether order changed.

		Custom per-instance buffers must follow self.entities when instance_order_version changes.
		Render components must each have one instance per entity.
		"""
		transforms, mesh_refs = world.get(transform_type), world.get(mesh_ref_type)
		version = (world, transform_type, mesh_ref_type, transforms.membership_version, mesh_refs.version("id", "shader_id"))
		if self._batch_version == version:
			self._sync_meshes()
			return False
		entities = world.where(transform_type, mesh_ref_type)
		refs = mesh_refs[entities]
		ordered, batches = build_batches(entities, refs.id, refs.shader_id)
		used_meshes = sorted({batch.mesh_id for batch in batches})
		mesh_rows = {mesh_id: i for i, mesh_id in enumerate(used_meshes)}
		for batch in batches:
			if batch.mesh_id not in self.meshes: raise KeyError(f"Unregistered mesh_id {batch.mesh_id}")
			if batch.shader_id not in self.shaders: raise KeyError(f"Unregistered shader_id {batch.shader_id}")
		metadata = np.zeros(len(used_meshes), dtype=mesh_metadata_dtype)
		for mesh_id, row in mesh_rows.items():
			info = self.meshes[mesh_id]
			metadata[row]["box_center"][:3] = info.box_center
			metadata[row]["box_extents"][:3] = info.box_extents
		params = np.zeros(len(batches), dtype=batch_dtype)
		commands = np.zeros(len(batches), dtype=indirect_dtype)
		group_counts = np.array([(batch.count + 63) // 64 for batch in batches], dtype=np.int64)
		workgroups = np.zeros(int(group_counts.sum()), dtype=workgroup_dtype)
		if len(batches):
			workgroups["batch_id"] = np.repeat(np.arange(len(batches)), group_counts)
			starts = np.cumsum(group_counts) - group_counts
			workgroups["instance_offset"] = (np.arange(len(workgroups)) - np.repeat(starts, group_counts)) * 64
		for batch in batches:
			mesh = self.meshes[batch.mesh_id].mesh
			params[batch.command_index] = (mesh_rows[batch.mesh_id], batch.offset, batch.count, 0)
			commands[batch.command_index] = (mesh.index_count, 0, mesh.index_range[0], mesh.vertex_range[0], batch.offset)
		for name in ("instances", "frustum_visible_instances", "main_visible_instances"):
			self._reserve_buffer(name, len(ordered))
		for name, values in (("mesh_metadata", metadata), ("batches", params), ("workgroups", workgroups), ("prepass_draw_cmd", commands), ("main_draw_cmd", commands)):
			self._upload_array(name, values)
		order_changed = not np.array_equal(self.entities, ordered)
		if order_changed: self.instance_order_version += 1
		self.entities, self.draw_batches = ordered, batches
		self._mesh_commands = {mesh_id: [] for mesh_id in used_meshes}
		for batch in batches:
			self._mesh_commands[batch.mesh_id].append(batch.command_index)
		self.mesh_rows = mesh_rows
		self._dirty_meshes.clear()
		self.workgroup_count = len(workgroups)
		self._batch_version = version
		return order_changed

	def refresh_bindings(self, hzb):
		cull_shader = self._cull_shader
		buffers = self.buffers
		if "cull" not in self.bindings:
			common = {name: buffers[name] for name in ("instances", "prepass_draw_cmd", "frustum_visible_instances", "mesh_metadata", "batches", "workgroups")}
			self.bindings["cull"] = cull_shader.bind_group(0, **common)
			self.bindings["frustum"] = cull_shader.bind_group(1, camera_params=self.camera_params_buffer)
			self.bindings["reset_prepass"] = cull_shader.bind_group(0, prepass_draw_cmd=buffers["prepass_draw_cmd"])
			self.bindings["reset_main"] = cull_shader.bind_group(1, camera_params=self.camera_params_buffer, main_draw_cmd=buffers["main_draw_cmd"])
		bindings = self.bindings.get("hiz")
		if bindings is None or bindings.resources["hzb_texture"] is not hzb.view:
			self.bindings["hiz"] = cull_shader.bind_group(1, camera_params=self.camera_params_buffer, hzb_texture=hzb.view, main_draw_cmd=buffers["main_draw_cmd"], main_visible_instances=buffers["main_visible_instances"])

	def reset_draw_counts(self, cmd, pipeline_name : str = "reset_draw_counts"):
		# Reset instance counters for indirect draw targets
		count = len(self.draw_batches)
		if not count: return
		groups = (count + 63) // 64
		x = min(groups, 65535)
		y = (groups + x - 1) // x
		if y > 65535: raise ValueError("Draw counter reset exceeds dispatch limits")

		pipeline = self.get_cull_pipeline(pipeline_name)

		with cmd.compute_pass(label=pipeline_name) as cp:
			cp.set_pipeline(pipeline)
			cp.set_bind_group(0, self.bindings["reset_prepass"])
			cp.set_bind_group(1, self.bindings["reset_main"])
			cp.dispatch(x, y)

	def cull_instances(self, cmd, stage : str = "frustum"):
		if not self.workgroup_count: return
		# Flatten a 2D dispatch so large scenes do not exceed the portable X limit.
		x = min(self.workgroup_count, 65535)
		y = (self.workgroup_count + x - 1) // x
		if y > 65535: raise ValueError("Culling workgroup table exceeds dispatch limits")

		pipeline_name = f"cull_{stage}"
		pipeline = self.get_cull_pipeline(pipeline_name) 

		with cmd.compute_pass(label=pipeline_name) as cp:
			cp.set_pipeline(pipeline)
			cp.set_bind_group(0, self.bindings["cull"])
			cp.set_bind_group(1, self.bindings[stage])
			cp.dispatch(x, y)

	def draw(self, rp, stage):
		commands = self.buffers[f"{stage}_draw_cmd"]
		for batch in self.draw_batches:
			spec = getattr(self.shaders[batch.shader_id], stage)
			rp.set_pipeline(spec.pipeline)
			for index, bindings in enumerate(self.bindings[batch.shader_id, stage]):
				rp.set_bind_group(index, bindings)
			for index, bindings in spec.bindings:
				rp.set_bind_group(index, bindings)
			mesh = self.meshes[batch.mesh_id].mesh
			rp.set_vertex_buffer(0, mesh.vertex_buffer)
			rp.set_index_buffer(mesh.index_buffer, format=mesh.index_format)
			rp.draw_indexed_indirect(commands, batch.command_index * indirect_dtype.itemsize)

	def invalidate_batches(self):
		"""Force grouping/command rebuilding on the next sync_batches call."""
		self._batch_version = None


class HZB:
	def __init__(self,
		shader : Shader|None = None,
		pipelines : tuple|None = None
	):
		self._shader = shader or Shader(filepath='scenes/shaders/hzb.shader', label="hzb")
		self._pipelines = pipelines or (
			ComputePipeline(self._shader, entry="reduce_depth", label="hzb_depth"),
			ComputePipeline(self._shader, entry="main", label="hzb")
		)
		self.size = ()
		self.depth_texture = None
		self.texture = None
		self.view = None
		self.passes = []

	def resize(self, size):
		shader, pipelines = self._shader, self._pipelines
		width, height = size
		if width <= 0 or height <= 0 or self.size == size: return False
		usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
		self.depth_texture = Texture(size, format="depth32float", usage=usage, label="prepass_depth")
		width, height = max(1, width // 2), max(1, height // 2)
		num_mips = max(width, height).bit_length()
		usage = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
		self.texture = Texture((width, height), format="r32float", usage=usage, mip_level_count=num_mips, label="hzb_pyramid")
		self.view = self.texture.view()
		views = [self.texture.view(base_mip_level=i, mip_level_count=1) for i in range(num_mips)]
		bindings = shader.bind_group(0, src_prepass=self.depth_texture.view(), dst_depth=views[0])
		self.passes = [(pipelines[0], bindings, width, height)]
		for mip in range(1, num_mips):
			width, height = max(1, width // 2), max(1, height // 2)
			bindings = shader.bind_group(0, src_depth=views[mip - 1], dst_depth=views[mip])
			self.passes.append((pipelines[1], bindings, width, height))
		self.size = size
		return True

	def build(self, cmd):
		for mip, (pipeline, bindings, width, height) in enumerate(self.passes):
			with cmd.compute_pass(label=f"hzb_mip_{mip}") as cp:
				cp.set_pipeline(pipeline)
				cp.set_bind_group(0, bindings)
				cp.dispatch((width + 15) // 16, (height + 15) // 16)


def build_batches(entities, mesh_ids, shader_ids):
	"""Pure CPU grouping; preserve source order inside each mesh/shader pair."""
	if not (entities.ndim == mesh_ids.ndim == shader_ids.ndim == 1 and len(entities) == len(mesh_ids) == len(shader_ids)):
		raise ValueError("Expected equally sized 1D entity, mesh ID, and shader ID arrays")
	order = np.lexsort((mesh_ids, shader_ids))
	mesh_ids, shader_ids = mesh_ids[order], shader_ids[order]
	changes = (mesh_ids[1:] != mesh_ids[:-1]) | (shader_ids[1:] != shader_ids[:-1])
	starts = np.r_[0, np.flatnonzero(changes) + 1] if len(order) else np.empty(0, dtype=int)
	ends = np.r_[starts[1:], len(order)] if len(order) else starts
	batches = [DrawBatch(int(mesh_ids[start]), int(shader_ids[start]), int(start), int(end - start), i) for i, (start, end) in enumerate(zip(starts, ends))]
	return entities[order].copy(), batches
