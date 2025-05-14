import numpy as np
import inspect
from typing import Type, Dict, Callable, Any, List, Tuple
from dataclasses import dataclass


# TODO: inherit int so we can distinguish between entities and ints ?
Entity = int
component = dataclass

# forward declaration
ComponentProxy = lambda s,e: None


class ComponentStorage:
    """
    Stores one component type using:
      - structured numpy array (_dense) for all fields
      - numpy arrays for sparse (entity -> head index)
      - optional multi-component support (contiguous blocks per entity)
    """
    NONE = -1
    entity_field = 'entity'

    def __init__(self,
                 component_cls: Type,
                 capacity: int = 128,
                 mult_comp: bool = False):
        self.component_cls = component_cls
        self.mult_comp     = mult_comp

        # Inspect dataclass annotations to build structured dtype
        ann = getattr(component_cls, "__annotations__", {})
        fields = [
            (name, np.float64) if typ is float
            else (name, int) if typ is int
            else (name, object)
            for name, typ in ann.items()
         ]

        # custom ndarray type field info
        # with a field for the entity associated to the component for book-keeping
        self._dtype = np.dtype( [(ComponentStorage.entity_field, int)] + fields)
        self._fields = ann.keys()

        # Expose object-field names for get_objects
        for i, (name, typ) in enumerate(ann.items()):
            if typ is not float:
                object.__setattr__(self, f"{name}_str", name)
                object.__setattr__(self, f"{name}_id", i)

        # buffers
        self._capacity  = capacity
        self._size      = 0
        self._sparse    = np.full(self._capacity, ComponentStorage.NONE, dtype=int)
        self._dense     = np.zeros(self._capacity, dtype=self._dtype)
        self._entities = ComponentStorage.NONE

    @property
    def capacity(self) -> int:
        return self._dense.shape[0]

    @property
    def sparse_size(self) -> int:
        return self._sparse.shape[0]

    @property
    def _entities(self) -> np.ndarray:
        return self._dense[ComponentStorage.entity_field]

    @_entities.setter
    def _entities(self, value:np.ndarray) -> None:
        self._dense[ComponentStorage.entity_field] = value

    def _grow_sparse(self, entity: int):
        old = self.sparse_size
        new_cap = max(old * 2, entity + 32)
        sp = np.full(new_cap, ComponentStorage.NONE, dtype=int)
        sp[:old] = self._sparse
        self._sparse = sp

    def _grow_dense(self, needed: int = 0):
        if self.mult_comp:
            valid = np.nonzero(self._entities[:self._size] != ComponentStorage.NONE)[0]
            compacted = self._dense[valid]
            new_size  = valid.size
        else:
            compacted = self._dense[:self._size]
            new_size  = self._size

        new_cap = max(self._capacity + new_size, needed+32)
        new_dense = np.zeros(new_cap, dtype=self._dtype)
        new_dense[ComponentStorage.entity_field] = ComponentStorage.NONE

        new_dense[:new_size] = compacted

        self._sparse = np.full(new_cap, ComponentStorage.NONE, dtype=int)
        for idx in range(new_size):
            ent = new_dense[ComponentStorage.entity_field][idx]
            if self._sparse[ent] == ComponentStorage.NONE:
                self._sparse[ent] = idx

        self._dense    = new_dense
        self._capacity = new_cap
        self._size     = new_size

    def _add(self, entity: int, component: Any) -> None:
        # ensure sparse space
        if entity >= self.sparse_size:
            self._grow_sparse(entity)
        # ensure dense space
        if self._size >= self.capacity:
            self._grow_dense()

        head = self._sparse[entity]
        if not self.mult_comp or head == ComponentStorage.NONE:
            # first or only
            idx = self._size
            self._sparse[entity] = idx
            self._size += 1
        else:
            # find end of current block
            idx = head
            while idx < self._size and self._entities[idx] == entity:
                idx += 1
            if idx == self._size:
                self._size += 1
            elif self._entities[idx] == ComponentStorage.NONE:
                pass
            else:
                length = idx - head
                self._relocate_to_end(entity, length)
                idx = self._size
                self._size += 1

        # write record at idx
        self._set(idx, entity, component)

    def _set(self, idx:int, entity:Entity, component:Any) -> None:
        self._entities[idx] = entity
        rec = self._dense[idx]
        for field in self._fields:
            rec[field] = getattr(component, field)

    def _relocate_to_end(self, entity: int, length: int):
        head = self._sparse[entity]
        if head == ComponentStorage.NONE or length <= 0:
            return

        old_size = self._size
        needed   = old_size + length
        if needed > self.capacity:
            self._grow_dense(needed)

        # copy block [head:head+length] → [old_size:old_size+length]
        self._dense[old_size:old_size+length] = self._dense[head:head+length]

        # mark holes
        self._entities[head:head+length] = ComponentStorage.NONE

        # update head
        self._sparse[entity] = old_size
        self._size = old_size + length

    # NOTE: bool return tells ecs to update entity_masks for this entity
    def _remove(self, component : ComponentProxy) -> bool:
        entity = component._entity
        head = self._sparse[entity]

        if head == ComponentStorage.NONE: return False

        if not self.mult_comp:
            # swap-pop last into head
            last = self._size - 1
            if head != last:
                self._dense[head] = self._dense[last]
                moved = self._entities[head]
                self._sparse[moved] = head
            self._sparse[entity] = ComponentStorage.NONE
            self._entities[last] = ComponentStorage.NONE
            self._size -= 1
            return True
        else:
            rem_idx = component._idx
            self._entities[rem_idx] = ComponentStorage.NONE
            # if we just removed the first component
            # set head to next component of the entity
            if head == rem_idx:
                nxt = rem_idx + 1
                while nxt < self._size and self._entities[nxt] == ComponentStorage.NONE:
                    nxt += 1
                # if there are none, tell ecs there are none left
                if self._entities[nxt] != entity: return True
                self._sparse[entity] = nxt
            return False

    def get(self, entity: int) -> Any:
        head = self._sparse[entity]

        if head == ComponentStorage.NONE:
            return [] if self.mult_comp else None

        if not self.mult_comp:
            return ComponentProxy(self, entity, head)

        proxies = []
        idx = head
        while idx < self._size and self._entities[idx] == entity:
            proxies.append(ComponentProxy(self, entity, idx))
            idx += 1
            while idx < self._size and self._entities[idx] == ComponentStorage.NONE:
                idx += 1
        return proxies

    def get_vector(self, *args) -> np.ndarray:
        """
        Usage:
            get_vector()                            # all component fields, for all entities
            get_vector(f1, f2, ...)                 # only f1,f2,..., for all entities
            get_vector(f1, f2, ..., entities_array) # f1,f2,... for given entities
        """
        # 1) Parse args
        entities = None
        fields = []
        if args:
            if isinstance(args[-1], np.ndarray):
                *fields, entities = args
                entities = np.atleast_1d(entities).astype(int)
            else:
                fields = list(args)

        # 2) Default to all fields (minus the entity column) if none specified
        if not fields:
            fields = self._fields

        if entities is None:
            return np.stack([ self._dense[field][:self._size] for field in fields ], axis=1)
        
        rows = self._sparse[entities]
        return np.stack([ self._dense[field][rows] for field in fields ], axis=1)

    def set_vector(self, entities: np.ndarray, vector: np.ndarray) -> None:
        """
        Overwrite the rows at `entities` for *all* component fields
        (except the internal 'entity' column) with the columns of `vector`.
        
        `vector` must be a 2-D array of shape (len(entities), Nfields),
        in the same order as get_vector() would return.
        """
        es  = np.atleast_1d(entities).astype(int)
        idx = self._sparse[es]

        arr = np.atleast_2d(vector)

        for j, field in enumerate(self._fields):
            self._dense[field][idx] = arr[:, j]



class ECS:
    """
    Fully optimized ECS with:
      - float-only numeric arrays for vectorized ops
      - preallocated component storage to avoid frequent reallocation
      - per-entity bitmask for fast component queries
      - entity ID recycling via free list
    """
    def __init__(self):
        self._stores: Dict[Type, ComponentStorage] = {}
        self._comp_bits: Dict[Type, int] = {}
        self._next_bit = 0
        self.entity_masks = np.zeros((0,), dtype=np.uint64)
        self._free_entities: List[Entity] = []
        self._next_entity_id = 0
    
    @property
    def entity_masks_size(self) -> int:
        return self.entity_masks.shape[0]
    
    def _grow_entity_mask(self, entity : Entity):
        entity_masks_size = self.entity_masks_size
        new_cap = max(entity_masks_size * 2, entity+32)
        new_entity_masks = np.full((new_cap,), 0, dtype=int)
        new_entity_masks[:entity_masks_size] = self.entity_masks[:entity_masks_size]
        self.entity_masks = new_entity_masks
    
    def create_entities(self, n: int) -> np.ndarray:
        """
        Create `n` new entity IDs, reusing deleted ones when possible.
        """
        out = []
        len_free = len(self._free_entities)
        if len_free:
            if len_free <= n:
                out = self._free_entities
                self._free_entities = []
                n -= len_free
            elif n == 1:
                return self._free_entities.pop()
            else:
                out = self._free_entities[-n:]
                self._free_entities[:-n]
                return out
        
        current_id = self._next_entity_id
        self._next_entity_id += n
        if self.entity_masks_size <= self._next_entity_id:
            self._grow_entity_mask(self._next_entity_id)
        out.extend( range(current_id, self._next_entity_id) )
        return out
        
    def create_entity(self, *components) -> int:
        [entity] = self.create_entities(1)
        self.add_component(entity, *components)
        return entity
    
    def delete_entity(self, entity: Entity) -> None:
        if entity in self._free_entities: return
        for comp_cls in list(self._comp_bits):
            if self.has_component(entity, comp_cls):
                self.remove_component(entity, comp_cls)
        self._free_entities.append(entity)
    
    def register(self, *component_types, allow_same_type_components_per_entity : bool = False, initial_capacity=128):
        for cls in component_types:
            if cls not in self._comp_bits:
                self._comp_bits[cls] = 1 << self._next_bit
                self._next_bit += 1
            if cls not in self._stores:
                self._stores[cls] = ComponentStorage(cls,
                    mult_comp=allow_same_type_components_per_entity,
                    capacity=128)
    
    def allow_multiple_components_per_entity(self, comp_cls: Type) -> None:
        store = self._stores.get(comp_cls)
        store.mult_comp = True
    
    def add_component(self, entity: Entity, *components) -> None:
        bits = self.entity_masks[entity]
        for comp in components:
            cls = type(comp)
            bit = self._comp_bits[cls]
            bits |= bit
            self._stores[cls]._add(entity, comp)
        self.entity_masks[entity] = bits

    def remove_component(self, component: ComponentProxy) -> None:
        if entity >= entity_masks_size: return
        store = self._stores.get(comp_cls)
        # _remove return true if there are no components of that type left in entity
        if store and store._remove(entity):
            bit = self._comp_bits.get(comp_cls, 0)
            self.entity_masks[entity] &= ~bit

    def get_store(self, comp_cls: Type) -> ComponentStorage:
        return self._stores.get(comp_cls)

    def get_vectors(self, *args):
        """
        returns numeric blocks of N component types associated with entities

        Usage:
            ecs.get_blocks(C1, C2, …, entities)

        - C1…Cn are component classes
        - entities: 1D array of entity IDs
        """
        *comp_clss, entities = args
        es = np.atleast_1d(entities).astype(int)
        return [self._stores.get(cls).get_vector(es) for cls in comp_clss]

    def where(self, *args) -> np.ndarray:
        """
        Usage:
          ecs.where(C1, C2, ..., predicate_fn)   # filter by predicate
          ecs.where(C1, C2, ...)                 # no predicate → all entities with those comps

        If the last argument is a callable, it’s used as a filter on the unpacked components;
        otherwise every entity that has all of the listed component types is returned.
        """

        # detect whether the last arg is a predicate or a component class
        last = args[-1]
        if callable(last) and not inspect.isclass(last) and len(args) >= 2:
            *comp_clss, predicate = args
        else:
            comp_clss = args
            predicate = None
        
        mask = 0
        for cls in comp_clss:
            mask |= self._comp_bits.get(cls, 0)
        if mask == 0:
            return []
        
        ents = np.nonzero((self.entity_masks & mask) == mask)[0]
        
        if predicate is None: return ents

        # apply predicate
        stores = tuple([self.get_store(cls) for cls in comp_clss])
        flags = [predicate(*[s.get(e) for s in stores]) for e in ents]
        return ents[np.array(flags, dtype=bool)]    



class ComponentProxy:
    """
    Lazy proxy for one component instance in a structured‐array storage.
    Wraps ComponentStorage, holds the entity ID and the record‐index in its _dense array.
    """
    def __init__(self, store: ComponentStorage, entity: Entity, dense_index: int):
        self._store = store
        self._idx = dense_index
        self._entity = entity
        self.__setattr__ = self.__setattr__impl

    def __getattr__(self, name: str):
        return self._store._dense[name][self._idx]

    def __setattr__impl(self, name: str, value: Any):
        self._store._dense[name][self.q] = value

    def __repr__(self):
        store = self._store
        idx = self._idx
        vals = {field: store._dense[field][idx] for field in store._fields }
        return f"<{store.component_cls.__name__}Proxy e={self._entity} {vals}>"

    def build(self) -> Any:
        """
        Reconstruct a full component instance (dataclass) from the proxy.
        """
        store = self._store
        kwargs = {}
        for field in store._fields:
            kwargs[field] = store._dense[field][self._idx]
        return store.component_cls(**kwargs)