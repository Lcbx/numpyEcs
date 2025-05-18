from pyray import *
from ecs import *

@component
class Position:
    x: float; y: float; z: float

@component
class Velocity:
    x: float; y: float; z: float

# @component
# class Orientation:
    # qx: float; qy: float; qz: float; qw: float

@component
class Mesh:
    color : Color
    
@component
class BoundingBox:
    x_min: float; y_min: float; z_min: float
    x_max: float; y_max: float; z_max: float

world = ECS()
world.register(Position, Velocity, Mesh, BoundingBox)

ground = world.create_entity(
    Position(0,-0.51,0),
    Mesh(LIGHTGRAY),
    BoundingBox(
        -25,0,-25,
        25,1,25
    )
)

rnd_uint8 = lambda : get_random_value(0, 255)
rnd_color = lambda : Color(rnd_uint8(),rnd_uint8(),rnd_uint8(),255)

SPACE_SIZE = 30
CUBE_MAX = 7

def add_cubes():
    for e in world.create_entities(250):
        world.add_component(e,
            Position(
                get_random_value(-SPACE_SIZE, SPACE_SIZE),
                get_random_value(0, 25),
                get_random_value(-SPACE_SIZE, SPACE_SIZE) ),
            Velocity(
                get_random_value(-4, 4),
                0,
                get_random_value(-4, 4) ),
            BoundingBox(
                get_random_value(-CUBE_MAX, 0), get_random_value(-CUBE_MAX, 0), get_random_value(-CUBE_MAX, 0),
                get_random_value(1, CUBE_MAX), get_random_value(1, CUBE_MAX), get_random_value(1, CUBE_MAX) ),
            Mesh(rnd_color()),
        )


camera = Camera3D(
    Vector3(30, 70,-25),
    Vector3(0,0,-25),
    Vector3(0,1,0),
    60.0,
    CAMERA_PERSPECTIVE
)


init_window(800, 450, "Hello")
set_target_fps(60)

positions = world.get_store(Position)
meshes = world.get_store(Mesh)
bboxes = world.get_store(BoundingBox)

frame_index = 0

while not window_should_close():
    
    frameTime = get_frame_time()

    pv = world.where(Position, Velocity)
    p_vec, v_vec = world.get_vectors(Position, Velocity, pv)
    p_vec += v_vec * frameTime
    positions.set_vector(pv, p_vec)
    
    begin_drawing()
    begin_mode_3d(camera)
    clear_background(WHITE)
    
    ents = world.where(Position, Mesh, BoundingBox)
    pos_arr, bb_arr = world.get_vectors(Position, BoundingBox, ents)

    for e, pos, bbox in zip(ents, pos_arr, bb_arr):
        bmin = bbox[:3]
        bmax = bbox[3:]
        size = bmax - bmin
        center = (bmax + bmin) * 0.5
        draw_cube(
            tuple(pos + center),
            size[0], # x
            size[1], # y
            size[2], # z
            meshes.get(e).color
        )

    frame_index+=1
    if frame_index%10 == 0 and get_fps() > 50: add_cubes()
    
    end_mode_3d()
    draw_text(f"fps {get_fps()} cubes {world.count} ", 10, 10, 20, LIGHTGRAY)
    end_drawing()
close_window()

