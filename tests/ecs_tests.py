import pytest

from ecs import *


def test_no_free_entities():
    ecs = ECS()
    ids = ecs.create_entities(3)
    assert isinstance(ids, list) or isinstance(ids, np.ndarray)
    assert list(ids) == [0, 1, 2]
    assert ecs._next_entity_id == 3
    assert ecs._free_entities == []

def test_single_free_entity_exact():
    ecs = ECS()
    ecs._free_entities = [10]
    [eid] = ecs.create_entities(1)
    assert isinstance(eid, Entity)
    assert eid == 10
    assert ecs._next_entity_id == 0
    assert ecs._free_entities == []

def test_single_free_entity_more_than_one():
    ecs = ECS()
    ecs._free_entities = [5]
    out = ecs.create_entities(3)
    assert out == [5, 0, 1]
    assert ecs._next_entity_id == 2
    assert ecs._free_entities == []

def test_multiple_delete_entity_twice():
    ecs = ECS()
    ecs._free_entities = [7, 8, 9, 10]
    ecs.delete_entity(7)
    assert ecs._free_entities == [7, 8, 9, 10]
    out = ecs.create_entities(2)
    assert out == [9,10]

def test_multiple_free_entities_exact_n():
    ecs = ECS()
    ecs._free_entities = [20, 30]
    ecs._next_entity_id = 5
    out = ecs.create_entities(2)
    assert out == [20, 30]
    assert ecs._free_entities == []
    assert ecs._next_entity_id == 5

def test_multiple_free_entities_more_than_n():
    ecs = ECS()
    ecs._free_entities = [100, 101, 102, 103]
    ecs._next_entity_id = 0
    out = ecs.create_entities(3)
    assert out == [101, 102, 103]
    assert ecs._free_entities[0] == 100
    assert ecs._next_entity_id == 0

@component
class Foo:
    a: float
    b: str

def test_single_component_storage_basic():
    s = ComponentStorage(Foo, mult_comp=False, capacity=2)
    assert s.get(1) is None
    
    s._add(1, Foo(1.23, "first"))
    proxy = s.get(1)
    assert proxy is not None
    assert pytest.approx(proxy.a) == 1.23
    assert proxy.b == "first"
    
    s._remove(proxy)
    assert s.get(1) is None
    s._add(1, Foo(4.56, "second"))
    proxy = s.get(1)
    assert pytest.approx(proxy.a) == 4.56
    assert proxy.b == "second"
    
    s._add(2, Foo(7.89, "third"))
    s._add(3, Foo(0.12, "fourth"))  # forces _grow_dense
    proxy = s.get(3)
    assert pytest.approx(proxy.a) == 0.12
    assert proxy.b == "fourth"

    # rebuild component (not useful, JIC)
    obj = proxy.build()
    assert pytest.approx(obj.a) == 0.12
    assert obj.b == "fourth"

    # check __setattr__ modifies the store 
    proxy.a = 0.69
    assert pytest.approx(s.get(3).a) == 0.69

    s._remove(s.get(2))
    assert s.get(2) is None

def test_single_component_storage_sparse_dense_integrity():
    s = ComponentStorage(Foo, mult_comp=False, capacity=1)
    
    for eid, val in enumerate([10.0, 20.0, 30.0], start=5):
        s._add(eid, Foo(val, f"val{eid}"))
    assert s._capacity >= 3
    
    for eid in (5,6,7):
        idx = s._sparse[eid]
        assert 0 <= idx < s._size
    
    foo6 = s.get(6)
    assert foo6 is not None
    
    s._remove(foo6)
    assert s.get(6) is None
    assert s.get(5) is not None and s.get(5).a == pytest.approx(10.0)
    assert s.get(7) is not None and s.get(7).a == pytest.approx(30.0)



def rotate_vectors_by_quaternions(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    q: shape (N,4)  as [qx,qy,qz,qw]
    v: shape (N,M,3) -- M vectors per entity
    returns: rotated vectors, same shape
    """
    q_vec = q[:, None, :3]             # (N,1,3)
    qw    = q[:, None, 3:4]            # (N,1,1)
    t = 2.0 * np.cross(q_vec, v, axis=2)  # (N,M,3)
    return v + qw * t + np.cross(q_vec, t, axis=2)


def update_world_aabb(ecs):
    
    # Non‐rotated boxes
    nr_ents = ecs.where(LocalAABB, Position)
    la, pos, wa = ecs.get_vectors(LocalAABB, Position, AxisAlignedBoundingBox, nr_ents)
    # just translate
    wa[:, :3] = la[:, :3] + pos
    wa[:, 3:] = la[:, 3:] + pos
    aabbs = ecs.get_store(AxisAlignedBoundingBox)
    aabbs.set_vector(nr_ents, wa)
    
    # Rotated boxes
    rot_ents = ecs.where(LocalAABB, Position, Orientation)
    la, pos, ori, wa = ecs.get_vectors(LocalAABB, Position, Orientation, AxisAlignedBoundingBox, rot_ents)
    # la:(N,6), pos:(N,3), ori:(N,4), wa:(N,6)
    mins = la[:, :3]
    maxs = la[:, 3:]
    # build 8 corners
    corners = np.array([[x,y,z] for x in (0,1)
                                 for y in (0,1)
                                 for z in (0,1)], int)
    offs = mins[:,None,:]*(1-corners)[None,:,:] + \
          maxs[:,None,:]*corners[None,:,:]
    # rotate & translate
    pts = rotate_vectors_by_quaternions(ori, offs) + pos[:,None,:]
    wa[:, :3] = pts.min(axis=1)
    wa[:, 3:] = pts.max(axis=1)
    aabbs.set_vector(rot_ents, wa)
    
    
def detect_aabb_overlaps(ecs):
    ents = ecs.where(AxisAlignedBoundingBox)
    blk = ecs.get_store(AxisAlignedBoundingBox).get_vector() # we want them all, so no need for indexing
    mins = blk[:, :3]
    maxs = blk[:, 3:]

    ok1 = mins[:,None,:] <= maxs[None,:,:]
    ok2 = maxs[:,None,:] >= mins[None,:,:]
    overlap = np.logical_and(ok1, ok2).all(axis=2)

    i,j = np.triu_indices(ents.size, k=1)
    hits = overlap[i,j]
    return [(int(ents[ii]), int(ents[jj]))
            for ii,jj,h in zip(i,j,hits) if h]

# Component definitions
@component
class Position2D:
    x: float; y: float

@component
class Velocity2D:
    vx: float; vy: float

@component
class Tag:
    id: int; name: str

@component
class Position:
    x: float; y: float; z: float

@component
class Orientation:
    qx: float; qy: float; qz: float; qw: float

@component
class LocalAABB:
    x_min: float; y_min: float; z_min: float
    x_max: float; y_max: float; z_max: float

@component
class AxisAlignedBoundingBox:
    x_min: float; y_min: float; z_min: float
    x_max: float; y_max: float; z_max: float

def test_movement_system():
    ecs = ECS()
    ecs.register(Position2D, Velocity2D, Tag)
    # Setup entities
    ecs.create_entity()
    ecs.create_entity(
        Position2D(0.0, 0.0),
        Velocity2D(1.0, 2.0))
    ecs.create_entity(
        Position2D(5.0, -3.0),
        Velocity2D(-1.0, 0.5),
        Tag(42, "Enemy"))

    # Verify initial positions
    positions = ecs.get_store(Position2D)
    p1 = positions.get(1)
    p2 = positions.get(2)
    assert (p1.x, p1.y) == pytest.approx((0.0, 0.0))
    assert (p2.x, p2.y) == pytest.approx((5.0, -3.0))

    # Apply movement only for Tag "Enemy"
    targets = ecs.where(Tag, Position2D, Velocity2D, lambda t, p, v: t.name == "Enemy")
    p_vec, v_vec = ecs.get_vectors(Position2D, Velocity2D, targets)
    p_vec -= v_vec * 0.5
    ecs.get_store(Position2D).set_vector(targets, p_vec)

    # Check updated position for entity 2
    p2_after = ecs.get_store(Position2D).get(2)
    assert (p2_after.x, p2_after.y) == pytest.approx((5.0 + 0.5, -3.0 - 0.25))

def test_aabb_system_and_overlap():
    ecs = ECS()
    ecs.register(LocalAABB, Position, AxisAlignedBoundingBox, Orientation)
    # Entity 1 with rotated AABB
    e1 = ecs.create_entity(
        LocalAABB(-1,-1,-1, 1,1,1),
        Position(5,0,0),
        AxisAlignedBoundingBox(0,0,0,0,0,0),
        Orientation(0,0,np.sin(np.pi/8), np.cos(np.pi/8))
    )
    # Entity 2 without rotation
    ecs.add_component(2,
        LocalAABB(-2,-2,-2, 2,2,2),
        Position(5,0,0),
        AxisAlignedBoundingBox(0,0,0,0,0,0))

    # Run systems
    update_world_aabb(ecs)
    wa1 = ecs.get_store(AxisAlignedBoundingBox).get(e1)
    # Compute expected rotated extents
    half_diag = np.sqrt(2)
    assert wa1.x_min == pytest.approx(5 - half_diag)
    assert wa1.x_max == pytest.approx(5 + half_diag)
    assert wa1.y_min == pytest.approx(-half_diag)
    assert wa1.y_max == pytest.approx(half_diag)
    assert wa1.z_min == pytest.approx(-1)
    assert wa1.z_max == pytest.approx(1)

    # Overlap detection
    hits = detect_aabb_overlaps(ecs)
    assert hits == [(e1, 2)]


@component
class MultiComp:
    val: float

def test_storage_multicomp_add_get_remove():
    # Create a storage that allows multiple MultiComp per entity
    store = ComponentStorage(MultiComp, mult_comp=True, capacity=4)
    entity_id = 42

    store._add(entity_id, MultiComp(1.))
    store._add(entity_id, MultiComp(2.))
    store._add(entity_id, MultiComp(3.))

    comps = store.get(entity_id)
    assert isinstance(comps, list)
    assert len(comps) == 3
    assert comps[0].val == pytest.approx(1.)
    assert comps[1].val == pytest.approx(2.)
    assert comps[2].val == pytest.approx(3.)

    idx = store._sparse[entity_id]
    block = store.get_vector().transpose()
    assert np.allclose(block, [1., 2., 3.])

    store._remove(store.get(entity_id)[1])
    comps = store.get(entity_id)
    assert isinstance(comps, list)
    assert len(comps) == 2
    assert comps[0].val == pytest.approx(1.0)
    assert comps[1].val == pytest.approx(3.)

def test_ecs_multicomp_reassignment():
    ecs = ECS()
    ecs.register(MultiComp, allow_same_type_components_per_entity=True, initial_capacity=5)
    store = ecs.get_store(MultiComp)

    ecs.create_entities(12)
    e = ecs.create_entity(MultiComp(1.1), MultiComp(2.2), MultiComp(3.3))
    ecs.create_entity(MultiComp(4.4))
    ecs.add_component(e, MultiComp(5.5))
    proxies = store.get(e)
    assert len(proxies) == 4
    store._remove(store.get(e)[1]) # 2.2
    store._remove(store.get(e)[1]) # 3.3
    proxies = store.get(e)
    assert len(proxies) == 2
    assert proxies[0].val == pytest.approx(1.1)
    assert proxies[1].val == pytest.approx(5.5)
    
    ecs.add_component(e, MultiComp(6.6))
    proxies = store.get(e)
    assert len(proxies) == 3
    assert proxies[0].val == pytest.approx(1.1)
    assert proxies[1].val == pytest.approx(6.6)
    assert proxies[2].val == pytest.approx(5.5)
    
    ecs.add_component(e, MultiComp(7.7))
    proxies = store.get(e)
    assert len(proxies) == 4
    assert proxies[0].val == pytest.approx(1.1)
    assert proxies[1].val == pytest.approx(6.6)
    assert proxies[2].val == pytest.approx(7.7)
    assert proxies[3].val == pytest.approx(5.5)