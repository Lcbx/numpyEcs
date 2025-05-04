from ecs import *
import pytest

@component
class Foo:
    a: float
    b: str

def test_single_component_storage_basic():
    s = ComponentStorage(Foo, mult_comp=False, initial_capacity=2)
    assert s.get(1) is None
    
    s.add(1, Foo(1.23, "first"))
    proxy1 = s.get(1)
    assert proxy1 is not None
    assert pytest.approx(proxy1.a) == 1.23
    assert proxy1.b == "first"
    
    s.remove(1)
    assert s.get(1) is None
    s.add(1, Foo(4.56, "second"))
    proxy2 = s.get(1)
    assert pytest.approx(proxy2.a) == 4.56
    assert proxy2.b == "second"
    
    s.add(2, Foo(7.89, "third"))
    s.add(3, Foo(0.12, "fourth"))  # forces _grow_dense
    proxy3 = s.get(3)
    assert pytest.approx(proxy3.a) == 0.12
    assert proxy3.b == "fourth"

    s.remove(2)
    assert s.get(2) is None

def test_single_component_storage_sparse_dense_integrity():
    s = ComponentStorage(Foo, mult_comp=False, initial_capacity=1)
    
    for eid, val in enumerate([10.0, 20.0, 30.0], start=5):
        s.add(eid, Foo(val, f"val{eid}"))
    assert s._capacity >= 3
    
    for eid in (5,6,7):
        idx = s.sparse[eid]
        assert 0 <= idx < s._size
    
    foo6 = s.get(6)
    assert foo6 is not None
    
    s.remove(6)
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
    ecs.set_vector(AxisAlignedBoundingBox, nr_ents, wa)
    
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
    ecs.set_vector(AxisAlignedBoundingBox, rot_ents, wa)
    
    
def detect_aabb_overlaps(ecs):
    ents = ecs.where(AxisAlignedBoundingBox)
    blk = ecs.get_vector(AxisAlignedBoundingBox, ents)
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
    # Setup entities
    ecs.add_component(1, Position2D(0.0, 0.0))
    ecs.add_component(1, Velocity2D(1.0, 2.0))
    ecs.add_component(2, Position2D(5.0, -3.0))
    ecs.add_component(2, Velocity2D(-1.0, 0.5))
    ecs.add_component(2, Tag(42, "Enemy"))

    # Verify initial positions
    p1 = ecs.get_component(1, Position2D)
    p2 = ecs.get_component(2, Position2D)
    assert (p1.x, p1.y) == pytest.approx((0.0, 0.0))
    assert (p2.x, p2.y) == pytest.approx((5.0, -3.0))

    # Apply movement only for Tag "Enemy"
    targets = ecs.where(Tag, Position2D, Velocity2D, lambda t, p, v: t.name == "Enemy")
    p_vec, v_vec = ecs.get_vectors(Position2D, Velocity2D, targets)
    p_vec -= v_vec * 0.5
    ecs.set_vector(Position2D, targets, p_vec)

    # Check updated position for entity 2
    p2_after = ecs.get_component(2, Position2D)
    assert (p2_after.x, p2_after.y) == pytest.approx((5.0 + 0.5, -3.0 - 0.25))

def test_aabb_system_and_overlap():
    ecs = ECS()
    # Entity 1 with rotated AABB
    e1 = ecs.create_entity(
        LocalAABB(-1,-1,-1, 1,1,1),
        Position(5,0,0),
        AxisAlignedBoundingBox(0,0,0,0,0,0),
        Orientation(0,0,np.sin(np.pi/8), np.cos(np.pi/8))
    )
    # Entity 2 without rotation
    ecs.add_component(2, LocalAABB(-2,-2,-2, 2,2,2))
    ecs.add_component(2, Position(5,0,0))
    ecs.add_component(2, AxisAlignedBoundingBox(0,0,0,0,0,0))

    # Run systems
    update_world_aabb(ecs)
    wa1 = ecs.get_component(e1, AxisAlignedBoundingBox)
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
    s = ComponentStorage(MultiComp, mult_comp=True, initial_capacity=4)
    entity_id = 42

    s.add(entity_id, MultiComp(1.))
    s.add(entity_id, MultiComp(2.))
    s.add(entity_id, MultiComp(3.))

    comps = s.get(entity_id)
    assert isinstance(comps, list)
    assert len(comps) == 3
    assert comps[0].val == pytest.approx(1.)
    assert comps[1].val == pytest.approx(2.)
    assert comps[2].val == pytest.approx(3.)

    idx = s.sparse[entity_id]
    block = s._nums[:s._size].transpose()
    assert np.allclose(block, [1., 2., 3.])

    s.remove(entity_id, which=1)
    comps = s.get(entity_id)
    assert isinstance(comps, list)
    assert len(comps) == 2
    assert comps[0].val == pytest.approx(1.0)
    assert comps[1].val == pytest.approx(3.)

def test_ecs_multicomp_reassignment():
    ecs = ECS()
    # Override the store for MultiComp to enable mult_comp
    s = ComponentStorage(MultiComp, mult_comp=True, initial_capacity=5)
    ecs._stores[MultiComp] = s

    ecs.create_entities(12)
    e = ecs.create_entity(MultiComp(1.1), MultiComp(2.2), MultiComp(3.3))
    ecs.create_entity(MultiComp(4.4))
    ecs.add_component(e, MultiComp(5.5))
    proxies = ecs.get_component(e, MultiComp)
    assert len(proxies) == 4
    s.remove(e, 1) # 2.2
    s.remove(e, 1) # 3.3
    proxies = ecs.get_component(e, MultiComp)
    assert len(proxies) == 2
    assert proxies[0].val == pytest.approx(1.1)
    assert proxies[1].val == pytest.approx(5.5)
    
    ecs.add_component(e, MultiComp(6.6))
    proxies = ecs.get_component(e, MultiComp)
    assert len(proxies) == 3
    assert proxies[0].val == pytest.approx(1.1)
    assert proxies[1].val == pytest.approx(6.6)
    assert proxies[2].val == pytest.approx(5.5)
    
    ecs.add_component(e, MultiComp(7.7))
    proxies = ecs.get_component(e, MultiComp)
    store = ecs._stores[MultiComp]
    head = store.sparse[e]
    block = store._nums[head : head + 4, 0]
    assert np.allclose(block, [1.1, 6.6, 7.7, 5.5])

test_single_component_storage_basic()
test_single_component_storage_sparse_dense_integrity()

test_movement_system()
test_aabb_system_and_overlap()

test_storage_multicomp_add_get_remove()
test_ecs_multicomp_reassignment()
