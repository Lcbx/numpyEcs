from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from inspect import signature
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence, TypeAlias, TypeVar, get_origin, get_type_hints

import numpy as np
from numpy.typing import NDArray

from common import higher_pow2

# Public component decorator.
component = dataclass

C = TypeVar("C")

Entity = np.uint64
Index  = np.uint32
Mask   = np.uint64

EntityArray : TypeAlias = NDArray[np.uint64]
IndexArray  : TypeAlias = NDArray[np.uint32]
MaskArray   : TypeAlias = NDArray[np.uint64]
BoolArray   : TypeAlias = NDArray[np.bool_]
FieldArray  : TypeAlias = NDArray[Any]

EntityLike: TypeAlias = Entity | EntityArray | Sequence[int]
IndexLike : TypeAlias = Index  | IndexArray  | Sequence[int]

SPARSE_PAGE_SIZE = 1024
ENTITY_PAGE_SIZE = 1024
INITIAL_CAPACITY = Index(128)

NONE = Index(np.iinfo(Index).max)
NO_ENTITY = Entity(np.iinfo(Entity).max)
ALIVE_BIT = Mask(1)


def _as_entities(entities: EntityLike) -> EntityArray:
	arr = np.asarray(entities, dtype=Entity)
	if arr.ndim == 0:
		return arr.reshape(1)
	if arr.ndim != 1:
		raise ValueError("entities must be a scalar or a 1-D array")
	return arr


def _as_rows(rows: IndexLike) -> IndexArray:
	arr = np.asarray(rows, dtype=Index)
	if arr.ndim == 0:
		return arr.reshape(1)
	if arr.ndim != 1:
		raise ValueError("rows must be a scalar or a 1-D array")
	return arr


def _higher_pow2(value: int) -> int:
	if value <= 1:
		return 1
	return 1 << (value - 1).bit_length()


def _component_fields(component_cls: type[Any]) -> tuple[tuple[str, Any], ...]:
	hints = get_type_hints(component_cls)

	if is_dataclass(component_cls):
		return tuple(
			(field.name, hints.get(field.name, field.type))
			for field in dataclass_fields(component_cls)
			if get_origin(hints.get(field.name, field.type)) is not ClassVar
		)

	return tuple(
		(name, annotation)
		for name, annotation in hints.items()
		if get_origin(annotation) is not ClassVar
	)


def _dtype_for(annotation: Any) -> np.dtype[Any]:
	# np.dtype(str) is <U1 and silently truncates arbitrary strings.
	if annotation is str or annotation is bytes:
		return np.dtype(object)

	try:
		return np.dtype(annotation)
	except TypeError:
		return np.dtype(object)


class ComponentSelection:
	__slots__ = ("_store", "_rows")

	def __init__(self, store: ComponentStorage, rows: IndexArray) -> None:
		self._store = store
		self._rows = rows

	@property
	def rows(self) -> IndexArray:
		return self._rows

	def __len__(self) -> int:
		return int(self._rows.size)

	def __getattr__(self, name: str) -> FieldArray:
		try:
			array = self._store._dense[name]
		except KeyError:
			raise AttributeError(name) from None
		return array[self._rows]

	def __setattr__(self, name: str, value: Any) -> None:
		if name.startswith("_"):
			object.__setattr__(self, name, value)
			return

		try:
			array = self._store._dense[name]
		except KeyError:
			raise AttributeError(name) from None
		array[self._rows] = value


class ComponentAccessor:
	__slots__ = ("_ecs", "_component_cls", "_store")

	def __init__(
		self,
		ecs: ECS,
		component_cls: type[Any],
		store: ComponentStorage,
	) -> None:
		self._ecs = ecs
		self._component_cls = component_cls
		self._store = store

	def __getitem__(self, entities: EntityLike) -> ComponentSelection:
		entity_array = _as_entities(entities)
		self._ecs._require_alive(entity_array)
		return ComponentSelection(self._store, self._store._get_rows(entity_array))

	@property
	def rows(self) -> IndexArray:
		return self._store._active_rows()

	def query(
		self,
		condition: Callable[..., Any],
		entities: EntityLike | None = None,
	) -> IndexArray:
		if entities is None:
			rows = self._store._active_rows()
		else:
			entity_array = _as_entities(entities)
			self._ecs._require_alive(entity_array)
			rows = self._store._get_rows(entity_array)
		return self._store._query_rows(condition, rows)

	def remove(self, rows: IndexLike) -> None:
		emptied = self._store._remove_rows(_as_rows(rows))
		if emptied.size:
			self._ecs._clear_component_mask(emptied, self._component_cls)


class ComponentStorage:
	mult_comp = False

	def __init__(
		self,
		component_cls: type[Any],
		capacity: Index = INITIAL_CAPACITY,
	) -> None:
		self.component_cls = component_cls

		field_defs = _component_fields(component_cls)
		if not field_defs:
			raise TypeError(f"{component_cls.__name__} has no annotated component fields")

		self.fields = tuple(name for name, _ in field_defs)
		self._field_dtypes = {
			name: _dtype_for(annotation)
			for name, annotation in field_defs
		}

		self._capacity = max(1, int(capacity))
		self._size = 0

		self._dense: dict[str, FieldArray] = {
			"entity": np.full(self._capacity, NO_ENTITY, dtype=Entity)
		}
		for name, dtype in self._field_dtypes.items():
			array = np.empty(self._capacity, dtype=dtype)
			if dtype == np.dtype(object):
				array.fill(None)
			self._dense[name] = array

		self._sparse: dict[int, IndexArray] = {}

	# ---------- sparse ----------

	def _get_dense_indices(self, entities: EntityArray) -> IndexArray:
		result = np.full(entities.size, NONE, dtype=Index)
		if entities.size == 0:
			return result

		page_ids = entities // Entity(SPARSE_PAGE_SIZE)
		offsets = (entities % Entity(SPARSE_PAGE_SIZE)).astype(np.intp, copy=False)

		for raw_page_id in np.unique(page_ids):
			page = self._sparse.get(int(raw_page_id))
			if page is None:
				continue
			selected = page_ids == raw_page_id
			result[selected] = page[offsets[selected]]

		return result

	def _set_dense_indices(self, entities: EntityArray, rows: IndexArray) -> None:
		if entities.size != rows.size:
			raise ValueError("entities and rows must have the same length")
		if entities.size == 0:
			return

		page_ids = entities // Entity(SPARSE_PAGE_SIZE)
		offsets = (entities % Entity(SPARSE_PAGE_SIZE)).astype(np.intp, copy=False)

		for raw_page_id in np.unique(page_ids):
			page_id = int(raw_page_id)
			selected = page_ids == raw_page_id
			page = self._sparse.get(page_id)
			if page is None:
				page = np.full(SPARSE_PAGE_SIZE, NONE, dtype=Index)
				self._sparse[page_id] = page
			page[offsets[selected]] = rows[selected]

	def _clear_dense_indices(self, entities: EntityArray) -> None:
		if entities.size == 0:
			return

		page_ids = entities // Entity(SPARSE_PAGE_SIZE)
		offsets = (entities % Entity(SPARSE_PAGE_SIZE)).astype(np.intp, copy=False)

		for raw_page_id in np.unique(page_ids):
			page_id = int(raw_page_id)
			page = self._sparse.get(page_id)
			if page is None:
				continue

			selected = page_ids == raw_page_id
			page[offsets[selected]] = NONE
			if not np.any(page != NONE):
				del self._sparse[page_id]

	# ---------- dense ----------

	def _grow_dense(self, minimum_capacity: int) -> None:
		if minimum_capacity <= self._capacity:
			return

		new_capacity = max(_higher_pow2(minimum_capacity), self._capacity * 2)
		for name, old in tuple(self._dense.items()):
			new = np.empty(new_capacity, dtype=old.dtype)
			new[: self._capacity] = old

			if name == "entity":
				new[self._capacity :] = NO_ENTITY
			elif old.dtype == np.dtype(object):
				new[self._capacity :] = None

			self._dense[name] = new

		self._capacity = new_capacity

	def _clear_rows(self, rows: IndexArray) -> None:
		if rows.size == 0:
			return

		self._dense["entity"][rows] = NO_ENTITY
		for field in self.fields:
			array = self._dense[field]
			if array.dtype == np.dtype(object):
				array[rows] = None

	def _active_rows(self) -> IndexArray:
		return np.arange(self._size, dtype=Index)

	def _get_rows(self, entities: EntityArray) -> IndexArray:
		rows = self._get_dense_indices(entities)
		if np.any(rows == NONE):
			missing = entities[rows == NONE]
			raise KeyError(f"entities do not have {self.component_cls.__name__}: {missing}")
		return rows

	# ---------- values ----------

	def _broadcast_field(
		self,
		field: str,
		value: Any,
		quantity: int,
		*,
		scalar: bool = False,
	) -> FieldArray:
		dtype = self._field_dtypes[field]

		if scalar:
			if dtype == np.dtype(object):
				array = np.empty(quantity, dtype=object)
				array.fill(value)
				return array
			return np.full(quantity, value, dtype=dtype)

		# Object-valued fields need care: arbitrary objects may themselves be iterable.
		if dtype == np.dtype(object):
			if isinstance(value, np.ndarray) and value.ndim == 1:
				if value.size == quantity:
					return value.astype(object, copy=False)
				if value.size == 1:
					array = np.empty(quantity, dtype=object)
					array.fill(value[0])
					return array
			elif isinstance(value, (list, tuple)):
				if len(value) == quantity:
					array = np.empty(quantity, dtype=object)
					for i, item in enumerate(value):
						array[i] = item
					return array
				if len(value) == 1:
					array = np.empty(quantity, dtype=object)
					array.fill(value[0])
					return array
			else:
				array = np.empty(quantity, dtype=object)
				array.fill(value)
				return array

			size = len(value)
			raise ValueError(
				f"field {field!r} has {size} values for {quantity} entities"
			)

		array = np.asarray(value, dtype=dtype)
		if array.ndim == 0:
			return np.full(quantity, array.item(), dtype=dtype)
		if array.ndim != 1:
			raise ValueError(f"field {field!r} must be scalar or 1-D")
		if array.size == quantity:
			return array
		if array.size == 1:
			return np.full(quantity, array[0], dtype=dtype)

		raise ValueError(
			f"field {field!r} has {array.size} values for {quantity} entities"
		)

	def _normalise_values(
		self,
		entities: EntityArray,
		values: object,
	) -> dict[str, FieldArray]:
		quantity = int(entities.size)

		# A component instance always means one scalar component value broadcast
		# over the whole entity range, even if one of its fields is iterable.
		if isinstance(values, self.component_cls):
			return {
				field: self._broadcast_field(
					field,
					getattr(values, field),
					quantity,
					scalar=True,
				)
				for field in self.fields
			}

		if len(self.fields) == 1:
			value = values[0] if isinstance(values, tuple) and len(values) == 1 else values
			field = self.fields[0]
			return {field: self._broadcast_field(field, value, quantity)}

		if isinstance(values, tuple):
			if len(values) != len(self.fields):
				raise ValueError(
					f"{self.component_cls.__name__} expects {len(self.fields)} fields, "
					f"got {len(values)}"
				)
			return {
				field: self._broadcast_field(field, value, quantity)
				for field, value in zip(self.fields, values)
			}

		packed = np.asarray(values)
		if packed.ndim == 1 and packed.shape[0] == len(self.fields):
			return {
				field: self._broadcast_field(field, packed[i], quantity)
				for i, field in enumerate(self.fields)
			}
		if packed.ndim == 2 and packed.shape == (quantity, len(self.fields)):
			return {
				field: self._broadcast_field(field, packed[:, i], quantity)
				for i, field in enumerate(self.fields)
			}

		raise ValueError(
			f"cannot map values with shape {packed.shape} to "
			f"{self.component_cls.__name__}{self.fields}"
		)

	# ---------- mutation ----------

	def _add_range(self, entities: EntityArray, values: object) -> None:
		if entities.size == 0:
			return
		if np.unique(entities).size != entities.size:
			raise ValueError(
				"duplicate entities are not allowed for a single component storage"
			)

		existing = self._get_dense_indices(entities)
		if np.any(existing != NONE):
			duplicate = entities[existing != NONE]
			raise ValueError(
				f"entities already have {self.component_cls.__name__}: {duplicate}"
			)

		field_values = self._normalise_values(entities, values)
		start = self._size
		stop = start + int(entities.size)
		self._grow_dense(stop)

		rows = np.arange(start, stop, dtype=Index)
		self._dense["entity"][start:stop] = entities
		for field, field_array in field_values.items():
			self._dense[field][start:stop] = field_array

		self._set_dense_indices(entities, rows)
		self._size = stop

	def _remove_entities(self, entities: EntityArray) -> EntityArray:
		if entities.size == 0:
			return np.empty(0, dtype=Entity)
		return self._remove_rows(self._get_rows(np.unique(entities)))

	def _remove_rows(self, rows: IndexArray) -> EntityArray:
		if rows.size == 0:
			return np.empty(0, dtype=Entity)

		rows = np.unique(rows)
		if np.any(rows >= self._size):
			raise IndexError("component row out of bounds")

		removed_entities = self._dense["entity"][rows].astype(Entity, copy=True)
		if np.any(removed_entities == NO_ENTITY):
			raise IndexError("component row is not active")

		old_size = self._size
		new_size = old_size - int(rows.size)

		removed_mask = np.zeros(old_size, dtype=np.bool_)
		removed_mask[rows.astype(np.intp)] = True

		holes = rows[rows < new_size]
		tail = np.arange(new_size, old_size, dtype=Index)
		sources = tail[~removed_mask[tail.astype(np.intp)]]

		self._clear_dense_indices(removed_entities)

		if holes.size:
			moved_entities = self._dense["entity"][sources].astype(Entity, copy=True)
			for array in self._dense.values():
				array[holes] = array[sources]
			self._set_dense_indices(moved_entities, holes)

		self._clear_rows(np.arange(new_size, old_size, dtype=Index))
		self._size = new_size
		return removed_entities

	# ---------- query ----------

	def _query_rows(
		self,
		condition: Callable[..., Any],
		rows: IndexArray,
	) -> IndexArray:
		if rows.size == 0:
			return np.empty(0, dtype=Index)

		params = signature(condition).parameters
		missing = [name for name in params if name not in self._dense]
		if missing:
			raise TypeError(
				f"query parameters are not component fields: {', '.join(missing)}"
			)

		result = np.asarray(
			condition(**{name: self._dense[name][rows] for name in params}),
			dtype=np.bool_,
		)

		if result.ndim == 0:
			return rows.copy() if bool(result) else np.empty(0, dtype=Index)
		if result.ndim != 1 or result.size != rows.size:
			raise ValueError("query condition must return one boolean per component row")
		return rows[result]


class MultiComponentStorage(ComponentStorage):
	mult_comp = True

	def __init__(
		self,
		component_cls: type[Any],
		capacity: Index = INITIAL_CAPACITY,
	) -> None:
		super().__init__(component_cls, capacity)
		self._count: dict[Entity, int] = {}
		self._live_count = 0

	def _active_rows(self) -> IndexArray:
		if self._size == 0:
			return np.empty(0, dtype=Index)
		return np.flatnonzero(
			self._dense["entity"][: self._size] != NO_ENTITY
		).astype(Index, copy=False)

	def _get_rows(self, entities: EntityArray) -> IndexArray:
		if entities.size == 0:
			return np.empty(0, dtype=Index)

		heads = self._get_dense_indices(entities)
		if np.any(heads == NONE):
			missing = entities[heads == NONE]
			raise KeyError(f"entities do not have {self.component_cls.__name__}: {missing}")

		blocks: list[IndexArray] = []
		for entity, head in zip(entities, heads):
			count = self._count.get(Entity(entity))
			if count is None:
				raise RuntimeError("multi-component sparse/count state is inconsistent")
			blocks.append(np.arange(int(head), int(head) + count, dtype=Index))

		return np.concatenate(blocks) if blocks else np.empty(0, dtype=Index)

	def _trim_tail(self) -> None:
		while self._size and self._dense["entity"][self._size - 1] == NO_ENTITY:
			self._size -= 1

	def _defragment(self) -> None:
		if self._live_count == self._size:
			return

		old_size = self._size
		if self._live_count == 0:
			self._clear_rows(np.arange(old_size, dtype=Index))
			self._size = 0
			self._sparse.clear()
			return

		rows = self._active_rows()
		live = self._live_count

		for array in self._dense.values():
			array[:live] = array[rows]
		self._clear_rows(np.arange(live, old_size, dtype=Index))
		self._size = live

		self._sparse.clear()
		entities = self._dense["entity"][:live].astype(Entity, copy=False)
		unique_entities, first = np.unique(entities, return_index=True)
		order = np.argsort(first)
		self._set_dense_indices(
			unique_entities[order],
			first[order].astype(Index, copy=False),
		)

	def _ensure_append_capacity(self, quantity: int) -> None:
		required = self._size + quantity
		if required <= self._capacity:
			return

		if self._live_count + quantity <= self._capacity:
			self._defragment()
			required = self._size + quantity

		self._grow_dense(required)

	def _write_block(
		self,
		start: int,
		entities: EntityArray,
		values: Mapping[str, FieldArray],
	) -> None:
		stop = start + int(entities.size)
		self._dense["entity"][start:stop] = entities
		for field, field_values in values.items():
			self._dense[field][start:stop] = field_values

	def _add_range(self, entities: EntityArray, values: object) -> None:
		if entities.size == 0:
			return

		field_values = self._normalise_values(entities, values)

		# Preserve first-seen owner order while allowing duplicate entities.
		owners, first = np.unique(entities, return_index=True)
		owners = owners[np.argsort(first)]

		for owner in owners:
			selected = entities == owner
			added_count = int(np.count_nonzero(selected))
			added_values = {
				field: array[selected]
				for field, array in field_values.items()
			}
			owner_one = np.asarray([owner], dtype=Entity)
			owner_array = np.full(added_count, owner, dtype=Entity)

			old_head = self._get_dense_indices(owner_one)[0]

			# New owner: append one contiguous block.
			if old_head == NONE:
				self._ensure_append_capacity(added_count)
				start = self._size
				self._write_block(start, owner_array, added_values)
				self._set_dense_indices(
					owner_one,
					np.asarray([start], dtype=Index),
				)
				self._count[Entity(owner)] = added_count
				self._size += added_count
				self._live_count += added_count
				continue

			old_count = self._count[Entity(owner)]
			head = int(old_head)
			extension_start = head + old_count
			extension_stop = extension_start + added_count

			# Extend in place if the following rows are free. If extending would
			# require growth while enough total free space exists, relocate instead
			# so defragmentation can reuse that space.
			inside_stop = min(extension_stop, self._size)
			following_free = (
				extension_start >= inside_stop
				or bool(
					np.all(
						self._dense["entity"][extension_start:inside_stop]
						== NO_ENTITY
					)
				)
			)
			growth_is_unavoidable = self._live_count + added_count > self._capacity
			can_extend = following_free and (
				extension_stop <= self._capacity or growth_is_unavoidable
			)

			if can_extend:
				self._grow_dense(extension_stop)
				self._write_block(extension_start, owner_array, added_values)
				self._size = max(self._size, extension_stop)
				self._count[Entity(owner)] = old_count + added_count
				self._live_count += added_count
				continue

			# Relocate the owner's whole block. Invalidate the old block before
			# capacity handling so defragmentation does not count it twice.
			old_rows = np.arange(head, head + old_count, dtype=Index)
			old_values = {
				field: self._dense[field][old_rows].copy()
				for field in self.fields
			}

			self._clear_rows(old_rows)
			self._clear_dense_indices(owner_one)
			self._live_count -= old_count
			self._trim_tail()

			block_size = old_count + added_count
			self._ensure_append_capacity(block_size)

			start = self._size
			stop_old = start + old_count
			stop = stop_old + added_count

			self._dense["entity"][start:stop] = owner
			for field in self.fields:
				self._dense[field][start:stop_old] = old_values[field]
				self._dense[field][stop_old:stop] = added_values[field]

			self._set_dense_indices(
				owner_one,
				np.asarray([start], dtype=Index),
			)
			self._count[Entity(owner)] = block_size
			self._size = stop
			self._live_count += block_size

	def _remove_entities(self, entities: EntityArray) -> EntityArray:
		if entities.size == 0:
			return np.empty(0, dtype=Entity)
		return self._remove_rows(self._get_rows(np.unique(entities)))

	def _remove_rows(self, rows: IndexArray) -> EntityArray:
		if rows.size == 0:
			return np.empty(0, dtype=Entity)

		rows = np.unique(rows)
		if np.any(rows >= self._size):
			raise IndexError("component row out of bounds")

		owners = self._dense["entity"][rows].astype(Entity, copy=True)
		if np.any(owners == NO_ENTITY):
			raise IndexError("component row is not active")

		emptied: list[Entity] = []

		for owner in np.unique(owners):
			owner_one = np.asarray([owner], dtype=Entity)
			head = int(self._get_dense_indices(owner_one)[0])
			count = self._count[Entity(owner)]
			block = np.arange(head, head + count, dtype=Index)

			remove_for_owner = rows[owners == owner]
			keep = block[~np.isin(block, remove_for_owner, assume_unique=True)]
			new_count = int(keep.size)

			if new_count:
				for field in self.fields:
					self._dense[field][head : head + new_count] = self._dense[field][keep]
				self._dense["entity"][head : head + new_count] = owner
				self._clear_rows(
					np.arange(head + new_count, head + count, dtype=Index)
				)
				self._count[Entity(owner)] = new_count
			else:
				self._clear_rows(block)
				self._clear_dense_indices(owner_one)
				del self._count[Entity(owner)]
				emptied.append(Entity(owner))

			self._live_count -= count - new_count

		self._trim_tail()
		return np.asarray(emptied, dtype=Entity)


class ECS:
	def __init__(self) -> None:
		self._stores: dict[type[Any], ComponentStorage] = {}
		self._comp_bits: dict[type[Any], Mask] = {}

		self._entity_masks: dict[int, MaskArray] = {}
		self._next_entity_id = 0
		self._entity_count = 0

	def __len__(self) -> int:
		return self._entity_count

	# ---------- entity masks ----------

	def _get_masks(self, entities: EntityArray) -> MaskArray:
		result = np.zeros(entities.size, dtype=Mask)
		if entities.size == 0:
			return result

		page_ids = entities // Entity(ENTITY_PAGE_SIZE)
		offsets = (entities % Entity(ENTITY_PAGE_SIZE)).astype(np.intp, copy=False)

		for raw_page_id in np.unique(page_ids):
			page = self._entity_masks.get(int(raw_page_id))
			if page is None:
				continue
			selected = page_ids == raw_page_id
			result[selected] = page[offsets[selected]]

		return result

	def _set_masks(self, entities: EntityArray, masks: MaskArray) -> None:
		if entities.size != masks.size:
			raise ValueError("entities and masks must have the same length")
		if entities.size == 0:
			return

		page_ids = entities // Entity(ENTITY_PAGE_SIZE)
		offsets = (entities % Entity(ENTITY_PAGE_SIZE)).astype(np.intp, copy=False)

		for raw_page_id in np.unique(page_ids):
			page_id = int(raw_page_id)
			selected = page_ids == raw_page_id
			values = masks[selected]
			page = self._entity_masks.get(page_id)

			if page is None:
				if not np.any(values):
					continue
				page = np.zeros(ENTITY_PAGE_SIZE, dtype=Mask)
				self._entity_masks[page_id] = page

			page[offsets[selected]] = values
			if not np.any(page):
				del self._entity_masks[page_id]

	def _set_bits(self, entities: EntityArray, bits: Mask) -> None:
		self._set_masks(entities, self._get_masks(entities) | bits)

	def _clear_bits(self, entities: EntityArray, bits: Mask) -> None:
		self._set_masks(entities, self._get_masks(entities) & ~bits)

	def _require_alive(self, entities: EntityArray) -> None:
		masks = self._get_masks(entities)
		dead = (masks & ALIVE_BIT) == 0
		if np.any(dead):
			raise KeyError(f"entities are not alive: {entities[dead]}")

	# ---------- component lookup ----------

	def _component_info(
		self,
		component_cls: type[Any],
	) -> tuple[ComponentStorage, Mask]:
		try:
			return self._stores[component_cls], self._comp_bits[component_cls]
		except KeyError:
			raise KeyError(
				f"component {component_cls.__name__} is not registered"
			) from None

	def _component_bits(self, component_types: Iterable[type[Any]]) -> Mask:
		bits = Mask(0)
		for component_cls in component_types:
			bits |= self._component_info(component_cls)[1]
		return bits

	def _clear_component_mask(
		self,
		entities: EntityArray,
		component_cls: type[Any],
	) -> None:
		self._clear_bits(entities, self._component_info(component_cls)[1])

	def _parse_add_components(
		self,
		components: tuple[object, ...],
	) -> list[tuple[type[Any], ComponentStorage, Mask, object]]:
		parsed: list[tuple[type[Any], ComponentStorage, Mask, object]] = []
		i = 0

		while i < len(components):
			item = components[i]

			if isinstance(item, type):
				component_cls = item
				store, bit = self._component_info(component_cls)

				if i + 1 >= len(components):
					raise TypeError(f"missing values for {component_cls.__name__}")

				values = components[i + 1]
				if isinstance(values, type) and values in self._stores:
					raise TypeError(f"missing values for {component_cls.__name__}")
				i += 2
			else:
				component_cls = type(item)
				store, bit = self._component_info(component_cls)
				values = item
				i += 1

			parsed.append((component_cls, store, bit, values))

		return parsed

	# ---------- registration ----------

	def register(
		self,
		*component_types: type[Any],
		allow_same_type_components_per_entity: bool = False,
		capacity: Index = INITIAL_CAPACITY,
	) -> None:
		storage_cls: type[ComponentStorage] = (
			MultiComponentStorage
			if allow_same_type_components_per_entity
			else ComponentStorage
		)

		for component_cls in component_types:
			if component_cls in self._stores:
				raise ValueError(f"{component_cls.__name__} is already registered")

			bit_index = len(self._comp_bits) + 1
			if bit_index >= 64:
				raise ValueError("uint64 entity masks support at most 63 component types")

			self._stores[component_cls] = storage_cls(component_cls, capacity)
			self._comp_bits[component_cls] = Mask(1 << bit_index)

	# ---------- entities ----------

	def create(self, quantity: int = 1) -> EntityArray:
		if quantity < 0:
			raise ValueError("quantity must be >= 0")
		if quantity == 0:
			return np.empty(0, dtype=Entity)

		if self._next_entity_id + quantity > int(NO_ENTITY):
			raise OverflowError("entity id space exhausted")

		start = self._next_entity_id
		entities = Entity(start) + np.arange(quantity, dtype=Entity)
		self._next_entity_id += quantity
		self._entity_count += quantity
		self._set_masks(entities, np.full(quantity, ALIVE_BIT, dtype=Mask))
		return entities

	def delete(self, entities: EntityLike) -> None:
		entity_array = np.unique(_as_entities(entities))
		if entity_array.size == 0:
			return

		self._require_alive(entity_array)
		masks = self._get_masks(entity_array)

		for component_cls, bit in self._comp_bits.items():
			selected = (masks & bit) != 0
			if np.any(selected):
				self._stores[component_cls]._remove_entities(entity_array[selected])

		self._set_masks(entity_array, np.zeros(entity_array.size, dtype=Mask))
		self._entity_count -= int(entity_array.size)

	# ---------- components ----------

	def add(self, entities: EntityLike, *components: object) -> None:
		entity_array = _as_entities(entities)
		if entity_array.size == 0 or not components:
			return

		self._require_alive(entity_array)
		parsed = self._parse_add_components(components)
		owners = np.unique(entity_array)

		for _component_cls, store, bit, values in parsed:
			store._add_range(entity_array, values)
			self._set_bits(owners, bit)

	def remove(self, entities: EntityLike, *component_types: type[Any]) -> None:
		entity_array = np.unique(_as_entities(entities))
		if entity_array.size == 0 or not component_types:
			return

		self._require_alive(entity_array)
		infos = [
			(component_cls, *self._component_info(component_cls))
			for component_cls in component_types
		]

		for _component_cls, store, bit in infos:
			store._remove_entities(entity_array)
			self._clear_bits(entity_array, bit)

	def get_store(self, component_cls: type[C]) -> ComponentStorage:
		return self._component_info(component_cls)[0]

	def get(self, component_cls: type[C]) -> ComponentAccessor:
		return ComponentAccessor(
			self,
			component_cls,
			self._component_info(component_cls)[0],
		)

	def where(
		self,
		*component_types: type[Any],
		exclude: type[Any] | Sequence[type[Any]] | None = None,
	) -> EntityArray:
		required = ALIVE_BIT | self._component_bits(component_types)

		if exclude is None:
			excluded = Mask(0)
		elif isinstance(exclude, type):
			excluded = self._component_bits((exclude,))
		else:
			excluded = self._component_bits(exclude)

		matches: list[EntityArray] = []
		for page_id in sorted(self._entity_masks):
			page = self._entity_masks[page_id]
			selected = (page & required) == required
			if excluded:
				selected &= (page & excluded) == 0

			offsets = np.flatnonzero(selected)
			if offsets.size:
				base = Entity(page_id * ENTITY_PAGE_SIZE)
				matches.append(base + offsets.astype(Entity, copy=False))

		return np.concatenate(matches) if matches else np.empty(0, dtype=Entity)
