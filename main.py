from ecs import *

@component
class Position2D:
    x: float
    y: float

@component
class Velocity2D:
    vx: float
    vy: float

@component
class Tag:
    id: int
    name: str

ecs = ECS()
# create entities:
ecs.add_component(1, Position2D(0.0, 0.0))
ecs.add_component(1, Velocity2D(1.0, 2.0))
ecs.add_component(2, Position2D(5.0, -3.0))
ecs.add_component(2, Velocity2D(-1.0, 0.5))
ecs.add_component(2, Tag(42, "Enemy"))

print(ecs.get_component(1, Position2D))
print(ecs.get_component(2, Position2D))

targets = ecs.where(Tag, Position2D, Velocity2D, lambda t, p, v: t.name == "Enemy" )
print("targets", targets)
p, v = ecs.get_vectors(Position2D, Velocity2D, targets)
p -= v * 0.5
ecs.set_vector(Position2D, targets, p)
print(ecs.get_component(2, Position2D))

@component
class Position:
    x: float
    y: float
    z: float

@component
class Orientation:
    qx: float
    qy: float
    qz: float
    qw: float

@component
class LocalAABB:
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

@component
class AxisAlignedBoundingBox:
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float


def rotate_vectors_by_quaternions(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    q: shape (N,4)  as [qx,qy,qz,qw]
    v: shape (N,M,3) -- M vectors per entity
    returns: rotated vectors, same shape
    """
    q_vec = q[:, None, :3]             # (N,1,3)
    qw    = q[:, None, 3:4]            # (N,1,1)
    # t = 2 * cross(q_vec, v)
    t = 2.0 * np.cross(q_vec, v, axis=2)  # (N,M,3)
    # v' = v + qw * t + cross(q_vec, t)
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


ecs = ECS()

ecs.create_entity(
    LocalAABB(-1,-1,-1, 1,1,1),
    Position(5,0,0),
    AxisAlignedBoundingBox(0,0,0,0,0,0),
    Orientation(0,0,np.sin(np.pi/8), np.cos(np.pi/8)) # 45° around Z axis
)

ecs.add_component(2, LocalAABB(-2,-2,-2, 2,2,2))
ecs.add_component(2, Position(5,0,0))
ecs.add_component(2, AxisAlignedBoundingBox(0,0,0,0,0,0))


update_world_aabb(ecs)
print(ecs.get_component(1, AxisAlignedBoundingBox))

hits = detect_aabb_overlaps(ecs)
print(hits) # [1, 2]