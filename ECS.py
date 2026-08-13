from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from enum import IntFlag
from inspect import signature
from typing import Any, Callable, ClassVar, Iterable, Iterator, Mapping, Sequence, TypeAlias, TypeVar, get_origin, get_type_hints

import numpy as np
from numpy.typing import NDArray

component = dataclass

Entity                  = np.uint64
Index                   = np.uint32
Generation              = np.uint32
Mask                    = np.uint64
ComponentPerEntityCount = np.uint16

EntityArray     : TypeAlias = NDArray[Entity]
IndexArray      : TypeAlias = NDArray[Index]
GenerationArray : TypeAlias = NDArray[Generation]
MaskArray       : TypeAlias = NDArray[Mask]
CountArray      : TypeAlias = NDArray[ComponentPerEntityCount]
BoolArray       : TypeAlias = NDArray[np.bool_]
FieldArray      : TypeAlias = NDArray[Any]

EntityLike: TypeAlias = Entity | int | EntityArray | Sequence[int]
IndexLike:  TypeAlias = Index  | int | IndexArray  | Sequence[int]


INDEX_MAX            = int(np.iinfo(Index).max)
INDEX_BITS           = np.dtype(Index).itemsize * 8
INDEX_MASK           = Entity(INDEX_MAX)
GENERATION_INCREMENT = Entity(1 << INDEX_BITS)
NONE                 = Index(INDEX_MAX)
NO_ENTITY            = Entity(np.iinfo(Entity).max)
INITIAL_CAPACITY     = 128


# useful for buffer resizing
# 0->1, 1->2, 2->4, 3->4, 4->8, 5->8
def higher_pow2(n: int|np.uint32|np.uint64) -> int:
	return 1 << int(n).bit_length()

def entity_index(entity: Entity | EntityArray) -> Index | IndexArray:
	result = np.asarray(entity, dtype=Entity)
	if result.ndim == 0: return Index(result & INDEX_MASK)
	return _entity_indices(result)


def entity_generation(entity: Entity | EntityArray) -> Generation | GenerationArray:
	result = np.asarray(entity, dtype=Entity) >> Entity(INDEX_BITS)
	if result.ndim == 0: return Generation(result)
	return result.astype(Generation, copy=False)


def _entity_indices(entities: EntityArray) -> IndexArray:
	return (entities & INDEX_MASK).astype(Index, copy=False)


def _as_entities(entities: EntityLike) -> EntityArray:
	array = np.asarray(entities, dtype=Entity)
	if array.ndim == 0:
		return array.reshape(1)
	if array.ndim != 1:
		raise ValueError("entities must be a scalar or a 1-D array")
	return array


def _as_rows(rows: IndexLike) -> IndexArray:
	array = np.asarray(rows, dtype=Index)
	if array.ndim == 0: return array.reshape(1)
	if array.ndim != 1: raise ValueError("rows must be a scalar or a 1-D array")
	return array


def _block_rows(heads: IndexArray, counts: CountArray) -> IndexArray:
	"""Concatenate arange(head, head + count) for each block, without Python loops."""
	if heads.size == 0: return np.empty(0, dtype=Index)

	counts64 = counts.astype(np.int64, copy=False)
	total = int(counts64.sum())
	if total == 0:
		return np.empty(0, dtype=Index)

	offsets = np.cumsum(counts64) - counts64
	rows = (
		np.repeat(heads.astype(np.int64, copy=False) - offsets, counts64)
		+ np.arange(total, dtype=np.int64)
	)
	return rows.astype(Index, copy=False)


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
	if annotation is str or annotation is bytes:
		return np.dtype(object)
	if isinstance(annotation, type) and issubclass(annotation, IntFlag):
		return np.dtype(np.int64)
	try:
		dtype = np.dtype(annotation)
	except TypeError:
		return np.dtype(object)
	if dtype.kind in ("U", "S"):
		return np.dtype(object)
	return dtype


class ComponentSelection:
	__slots__ = ("_store", "_rows")

	def __init__(self, store: ComponentStorage, rows: IndexArray) -> None:
		self._store = store
		self._rows = rows

	@property
	def rows(self) -> IndexArray:
		return self._rows

	def vector(self, *field_names: str) -> FieldArray:
		field_names = field_names or self._store.fields
		return np.column_stack( tuple(self[field] for field in field_names) )

	def __len__(self) -> int:
		return int(self._rows.size)

	def __getitem__(self, field: str) -> FieldArray:
		return self._store._dense[field][self._rows]

	def __setitem__(self, field: str, value: Any) -> None:
		self._store._dense[field][self._rows] = value

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
		entities_ = self._ecs._filter_alive(_as_entities(entities))
		indices = _entity_indices(entities_)
		return ComponentSelection(self._store, self._store._get_rows(indices))

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
			entities_ = self._ecs._filter_alive(np.unique(_as_entities(entities)))
			rows = self._store._get_rows(_entity_indices(entities_))
		return self._store._query_rows(condition, rows)

	def remove(self, rows: IndexLike) -> None:
		rows_ = np.unique(_as_rows(rows))
		rows_ = rows_[rows_ < self._store._size]
		rows_ = rows_[self._store._dense["entity"][rows_] != NO_ENTITY]
		if rows_.size == 0:
			return

		emptied = self._store._remove_rows(rows_)
		if emptied.size:
			self._ecs._clear_component_mask(_entity_indices(emptied), self._component_cls)


class ComponentStorage:

	def __init__(
		self,
		component_cls: type[Any],
		capacity: int = INITIAL_CAPACITY,
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
		self._sparse = np.full(self._capacity, NONE, dtype=Index)

		self._dense: dict[str, FieldArray] = {
			"entity": np.full(self._capacity, NO_ENTITY, dtype=Entity)
		}
		for name, dtype in self._field_dtypes.items():
			array = np.empty(self._capacity, dtype=dtype)
			if dtype == np.dtype(object):
				array.fill(None)
			self._dense[name] = array

	# ---------- sparse ----------

	def _grow_sparse(self, minimum_size: int) -> None:
		if minimum_size <= self._sparse.size:
			return
		new_size = max(higher_pow2(minimum_size), self._sparse.size * 2)
		sparse = np.full(new_size, NONE, dtype=Index)
		sparse[: self._sparse.size] = self._sparse
		self._sparse = sparse

	def _ensure_sparse(self, indices: IndexArray) -> None:
		if indices.size:
			self._grow_sparse(int(indices.max()) + 1)

	def _get_dense_indices(self, indices: IndexArray) -> IndexArray:
		rows = np.full(indices.size, NONE, dtype=Index)
		in_range = indices < self._sparse.size
		rows[in_range] = self._sparse[indices[in_range]]
		return rows

	def _set_dense_indices(self, indices: IndexArray, rows: IndexArray) -> None:
		self._ensure_sparse(indices)
		self._sparse[indices] = rows

	def _clear_dense_indices(self, indices: IndexArray) -> None:
		self._sparse[indices] = NONE

	# ---------- dense ----------

	def _grow_dense(self, minimum_capacity: int) -> None:
		if minimum_capacity <= self._capacity: return

		new_capacity = higher_pow2(minimum_capacity)
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
		self._dense["entity"][rows] = NO_ENTITY
		for field in self.fields:
			array = self._dense[field]
			if array.dtype == np.dtype(object):
				array[rows] = None

	def _active_rows(self) -> IndexArray:
		return np.arange(self._size, dtype=Index)

	def _get_rows(self, indices: IndexArray) -> IndexArray:
		rows = self._get_dense_indices(indices)
		return rows[rows != NONE]

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
					return np.fromiter(value, dtype=object, count=quantity)
				if len(value) == 1:
					array = np.empty(quantity, dtype=object)
					array.fill(value[0])
					return array
			else:
				array = np.empty(quantity, dtype=object)
				array.fill(value)
				return array

			raise ValueError(
				f"field {field!r} has {len(value)} values for {quantity} entities"
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

	def _add_normalised(
		self,
		entities: EntityArray,
		field_values: Mapping[str, FieldArray],
	) -> None:
		indices = _entity_indices(entities) 
		self._ensure_sparse(indices)

		count = int(entities.size)
		start = self._size
		stop = start + count
		self._grow_dense(stop)

		rows = np.arange(start, stop, dtype=Index)
		self._dense["entity"][start:stop] = entities
		for field, field_values_array in field_values.items():
			self._dense[field][start:stop] = field_values_array
		self._set_dense_indices(indices, rows)
		self._size = stop
	
	def _add_range(self, entities: EntityArray, values: object) -> None:
		self._add_normalised(entities, self._normalise_values(entities, values))


	def _remove_entities(self, indices: IndexArray) -> EntityArray:
		return self._remove_rows(self._get_rows(indices))

	def _remove_rows(self, rows: IndexArray) -> EntityArray:
		if rows.size == 0:
			return np.empty(0, dtype=Entity)

		removed_entities = self._dense["entity"][rows].astype(Entity, copy=True)
		old_size = self._size
		new_size = old_size - int(rows.size)

		removed_mask = np.zeros(old_size, dtype=np.bool_)
		removed_mask[rows] = True
		holes = rows[rows < new_size]
		tail = np.arange(new_size, old_size, dtype=Index)
		sources = tail[~removed_mask[tail]]

		self._clear_dense_indices(_entity_indices(removed_entities))

		if holes.size:
			moved_entities = self._dense["entity"][sources].astype(Entity, copy=True)
			for array in self._dense.values():
				array[holes] = array[sources]
			self._set_dense_indices(_entity_indices(moved_entities), holes)

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

	def __init__(
		self,
		component_cls: type[Any],
		capacity: int = INITIAL_CAPACITY,
	) -> None:
		super().__init__(component_cls, capacity)
		self._count = np.zeros(self._sparse.size, dtype=ComponentPerEntityCount)
		self._live_count = 0

	def _grow_sparse(self, minimum_size: int) -> None:
		old_size = self._sparse.size
		super()._grow_sparse(minimum_size)
		if self._sparse.size == old_size:
			return
		count = np.zeros(self._sparse.size, dtype=ComponentPerEntityCount)
		count[: self._count.size] = self._count
		self._count = count

	def _active_rows(self) -> IndexArray:
		return np.flatnonzero(
			self._dense["entity"][: self._size] != NO_ENTITY
		).astype(Index, copy=False)

	def _get_rows(self, indices: IndexArray) -> IndexArray:
		heads = self._get_dense_indices(indices)
		present = heads != NONE
		return _block_rows(heads[present], self._count[indices[present]])

	def _trim_tail(self) -> None:
		if self._live_count == 0:
			self._size = 0
			return
		if self._size == 0 or self._dense["entity"][self._size - 1] != NO_ENTITY:
			return
		reverse_live = self._dense["entity"][: self._size][::-1] != NO_ENTITY
		self._size -= int(np.argmax(reverse_live))

	def _defragment(self, required_capacity: int) -> None:
		rows = self._active_rows()
		live = int(rows.size)

		new_arrays = required_capacity > self._capacity
		new_capacity = (
			higher_pow2(required_capacity) if new_arrays
			else self._capacity
		)

		if live == self._size and new_capacity == self._capacity:
			return

		for name in self._dense:
			old = self._dense[name]
			new = np.empty(new_capacity, dtype=old.dtype) if new_arrays else old

			new[:live] = old[rows]

			if name == "entity":
				new[live:] = NO_ENTITY
			elif old.dtype == np.dtype(object):
				new[live:] = None

			self._dense[name] = new

		self._capacity = new_capacity
		self._size = live

		self._sparse.fill(NONE)

		owners = self._dense["entity"][:live]
		starts = np.flatnonzero(
			np.diff(owners, prepend=NO_ENTITY)
		).astype(Index, copy=False)

		self._set_dense_indices(_entity_indices(owners[starts]), starts)

	def _ensure_append_capacity(self, quantity: int) -> None:
		if self._size + quantity > self._capacity:
			self._defragment(self._live_count + quantity)

	def _add_normalised(
		self,
		entities: EntityArray,
		field_values: Mapping[str, FieldArray],
	) -> None:
		if entities.size == 0:
			return

		# Group new components by owner.
		order = np.argsort(entities, kind="stable")
		entities = entities[order]
		field_values = {
			field: values[order]
			for field, values in field_values.items()
		}

		starts = np.flatnonzero(
			np.r_[True, entities[1:] != entities[:-1]]
		)
		add_counts = np.diff(
			np.r_[starts, entities.size]
		).astype(ComponentPerEntityCount, copy=False)

		owners = entities[starts]
		owner_indices = _entity_indices(owners)

		self._ensure_sparse(owner_indices)

		old_heads = self._sparse[owner_indices]
		existing = old_heads != NONE

		old_counts = np.zeros_like(add_counts)
		old_counts[existing] = self._count[owner_indices[existing]]
		final_counts = old_counts + add_counts

		# Save old values from affected existing blocks.
		old_rows = _block_rows(
			old_heads[existing],
			old_counts[existing],
		)

		old_values = {
			field: self._dense[field][old_rows].copy()
			for field in self.fields
		}

		# Remove affected old blocks.
		self._clear_rows(old_rows)
		self._sparse[owner_indices[existing]] = NONE

		self._live_count -= old_rows.size
		self._trim_tail()

		# Append rebuilt blocks for all affected owners.
		append_count = int(final_counts.sum())
		self._ensure_append_capacity(append_count)

		start = self._size
		offsets = np.cumsum(final_counts, dtype=np.uint64) - final_counts
		new_heads = (start + offsets).astype(Index, copy=False)

		self._dense["entity"][start:start + append_count] = np.repeat(
			owners,
			final_counts,
		)

		old_destinations = _block_rows(
			new_heads,
			old_counts,
		)

		add_destinations = _block_rows(
			(new_heads + old_counts.astype(Index, copy=False)),
			add_counts,
		)

		for field in self.fields:
			self._dense[field][old_destinations] = old_values[field]
			self._dense[field][add_destinations] = field_values[field]

		self._sparse[owner_indices] = new_heads
		self._count[owner_indices] = final_counts

		self._size += append_count
		self._live_count += append_count

	def _remove_entities(self, indices: IndexArray) -> EntityArray:
		return self._remove_rows(np.sort(self._get_rows(indices)))

	def _remove_rows(self, rows: IndexArray) -> EntityArray:
		removed_owners = self._dense["entity"][rows]

		owners, _removed_counts = np.unique(removed_owners, return_counts=True)
		owner_indices = _entity_indices(owners)

		heads = self._sparse[owner_indices]
		old_counts = self._count[owner_indices]
		removed_counts = _removed_counts.astype(ComponentPerEntityCount, copy=False)
		new_counts = (old_counts - removed_counts).astype(ComponentPerEntityCount, copy=False)

		block_rows = _block_rows(heads, old_counts)

		# rows is sorted, so find which rows of the affected blocks are removed
		positions = np.searchsorted(rows, block_rows)
		removed = positions < rows.size
		removed[removed] &= rows[positions[removed]] == block_rows[removed]
		kept_rows = block_rows[~removed]
		destinations = _block_rows(heads, new_counts)

		# Advanced indexing materializes the RHS, so overlapping moves are safe.
		for field in self.fields:
			self._dense[field][destinations] = self._dense[field][kept_rows]
		self._dense["entity"][destinations] = np.repeat(owners, new_counts)

		# Clear what remains at the end of each compacted block.
		tail_heads = heads + new_counts.astype(Index, copy=False)
		tails = _block_rows(tail_heads, removed_counts)
		self._clear_rows(tails)

		self._count[owner_indices] = new_counts

		emptied = new_counts == 0
		self._sparse[owner_indices[emptied]] = NONE

		self._live_count -= rows.size
		self._trim_tail()

		return owners[emptied]


class ECS:
	def __init__(self) -> None:
		self._stores: dict[type[Any], ComponentStorage] = {}
		self._comp_bits: dict[type[Any], Mask] = {}

		self.entity_masks = np.zeros(0, dtype=Mask)
		self._current_entities = np.full(0, NO_ENTITY, dtype=Entity)
		self._free_entities: list[Entity] = []
		self._next_entity_index = 0
		self._entity_count = 0

	def __len__(self) -> int:
		return self._entity_count

	# ---------- entity state ----------

	def _grow_entity_state(self, minimum_size: int) -> None:
		if minimum_size <= self.entity_masks.size:
			return
		old_size = self.entity_masks.size
		new_size = max(higher_pow2(minimum_size), max(1, old_size * 2))

		masks = np.zeros(new_size, dtype=Mask)
		masks[:old_size] = self.entity_masks
		self.entity_masks = masks

		current = np.full(new_size, NO_ENTITY, dtype=Entity)
		current[:old_size] = self._current_entities
		self._current_entities = current

	def _alive_mask(self, entities: EntityArray) -> BoolArray:
		indices = _entity_indices(entities)
		alive = indices < self._current_entities.size

		alive[alive] &= (
			self._current_entities[indices[alive]]
			== entities[alive]
		)
		return alive

	def _filter_alive(self, entities: EntityArray) -> EntityArray:
		return entities[self._alive_mask(entities)]

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
		indices: IndexArray,
		component_cls: type[Any],
	) -> None:
		self.entity_masks[indices] &= ~self._component_info(component_cls)[1]

	def _parse_add_components(
		self,
		params: tuple[object, ...],
	) -> Iterator[tuple[type[Any], ComponentStorage, Mask, object]]:
		i = 0

		while i < len(params):
			item = params[i]

			if isinstance(item, type):
				component_cls = item
				store, bit = self._component_info(component_cls)

				if i + 1 >= len(params):
					raise TypeError(f"missing components for {component_cls.__name__}")

				components = params[i + 1]

				if isinstance(components, type) and components in self._stores:
					raise TypeError(f"missing components for {component_cls.__name__}")
				i += 2

			else:
				component_cls = type(item)
				store, bit = self._component_info(component_cls)
				components = item
				i += 1

			yield component_cls, store, bit, components

	# ---------- registration ----------

	def register(
		self,
		*component_types: type[Any],
		allow_same_type_components_per_entity: bool = False,
		capacity: int = INITIAL_CAPACITY,
	) -> None:
		storage_cls: type[ComponentStorage] = (
			MultiComponentStorage
			if allow_same_type_components_per_entity
			else ComponentStorage
		)
		for component_cls in component_types:
			if component_cls in self._stores:
				raise ValueError(f"{component_cls.__name__} is already registered")
			bit_index = len(self._comp_bits)
			if bit_index >= 64:
				raise ValueError("uint64 entity masks support at most 64 component types")
			self._stores[component_cls] = storage_cls(component_cls, capacity)
			self._comp_bits[component_cls] = Mask(1 << bit_index)

	# ---------- entities ----------

	def create(self, quantity: int = 1) -> EntityArray:
		if quantity < 0:
			raise ValueError("quantity must be >= 0")
		if quantity == 0:
			return np.empty(0, dtype=Entity)

		reuse_count = min(quantity, len(self._free_entities))
		reused = np.empty(0, dtype=Entity)

		if reuse_count:
			old = np.asarray(self._free_entities[-reuse_count:], dtype=Entity)
			del self._free_entities[-reuse_count:]
			reused = old + GENERATION_INCREMENT
			indices = _entity_indices(reused)
			self._current_entities[indices] = reused
			self.entity_masks[indices] = Mask(0)

		new_count = quantity - reuse_count
		if new_count == 0:
			self._entity_count += quantity
			return reused

		stop = self._next_entity_index + new_count
		if stop > INDEX_MAX:
			raise OverflowError("entity index space exhausted")
		self._grow_entity_state(stop)
		indices = np.arange(self._next_entity_index, stop, dtype=Index)
		fresh = indices.astype(Entity)
		self._current_entities[indices] = fresh
		self.entity_masks[indices] = Mask(0)
		self._next_entity_index = stop
		self._entity_count += quantity

		if reuse_count == 0:
			return fresh
		return np.concatenate((reused, fresh))

	def delete(self, entities: EntityLike) -> None:
		entity_array = self._filter_alive(_as_entities(entities))
		if entity_array.size == 0:
			return

		# Duplicate live IDs must not be recycled twice.
		entity_array = np.unique(entity_array)
		indices = _entity_indices(entity_array)
		masks = self.entity_masks[indices].copy()

		for component_cls, bit in self._comp_bits.items():
			selected = (masks & bit) != 0
			if np.any(selected):
				self._stores[component_cls]._remove_entities(indices[selected])

		self.entity_masks[indices] = Mask(0)
		self._current_entities[indices] = NO_ENTITY
		self._free_entities.extend(entity_array.tolist())
		self._entity_count -= int(entity_array.size)

	# ---------- components ----------

	def add(self, entities: EntityLike, *components: object) -> None:
		entity_array = _as_entities(entities)
		if entity_array.size == 0 or not components:
			return

		alive = self._alive_mask(entity_array)
		if not np.any(alive):
			return

		parsed = list(self._parse_add_components(components))
		live_entities = entity_array[alive]
		live_indices = _entity_indices(live_entities)

		for component_cls, store, bit, values in parsed:
			normalised = store._normalise_values(entity_array, values)
			filtered_values = {
				field: field_values[alive]
				for field, field_values in normalised.items()
			}

			store._add_normalised(live_entities, #live_indices,
				filtered_values)
			self.entity_masks[live_indices] |= bit

	def remove(self, entities: EntityLike, *component_types: type[Any]) -> None:
		entity_array = np.unique(self._filter_alive(_as_entities(entities)))
		if entity_array.size == 0 or not component_types:
			return

		indices = _entity_indices(entity_array)
		for component_cls in component_types:
			store, bit = self._component_info(component_cls)
			store._remove_entities(indices)
			self.entity_masks[indices] &= ~bit

	def get_store(self, component_cls: type) -> ComponentStorage:
		return self._component_info(component_cls)[0]

	def get(self, component_cls: type) -> ComponentAccessor:
		store, _ = self._component_info(component_cls)
		return ComponentAccessor(self, component_cls, store)

	def where(
		self,
		*component_types: type[Any],
		exclude: type[Any] | Sequence[type[Any]] | None = None,
	) -> EntityArray:
		if not component_types:
			return np.empty(0, dtype=Entity)

		required = self._component_bits(component_types)
		if exclude is None:
			excluded = Mask(0)
		elif isinstance(exclude, type):
			excluded = self._component_bits((exclude,))
		else:
			excluded = self._component_bits(exclude)

		masks = self.entity_masks[: self._next_entity_index]
		selected = (masks & required) == required
		if excluded: selected &= (masks & excluded) == 0

		indices = np.flatnonzero(selected).astype(Index, copy=False)
		return self._current_entities[indices].astype(Entity, copy=False)
