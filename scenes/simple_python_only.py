from pyray import *
from dataclasses import dataclass

@dataclass
class Cube:
    position: Vector3
    velocity: Vector3
    color : Color
    BoundingBox_min : Vector3
    BoundingBox_max : Vector3


ground = Cube(
    Vector3(0,-0.51,0),
    Vector3(0,0,0),
    LIGHTGRAY,
    Vector3(-25,0,-25),
    Vector3(25,1,25)
)

cubes = [ground]


rnd_uint8 = lambda : get_random_value(0, 255)
rnd_color = lambda : Color(rnd_uint8(),rnd_uint8(),rnd_uint8(),255)

SPACE_SIZE = 30
CUBE_MAX = 7

def add_cubes():
    for e in range(250):
        cubes.append(Cube(
            Vector3(
                get_random_value(-SPACE_SIZE, SPACE_SIZE),
                get_random_value(0, 25),
                get_random_value(-SPACE_SIZE, SPACE_SIZE) ),
            Vector3(
                get_random_value(-4, 4),
                0,
                get_random_value(-4, 4) ),
            rnd_color(),
            Vector3(get_random_value(-CUBE_MAX, 0), get_random_value(-CUBE_MAX, 0), get_random_value(-CUBE_MAX, 0) ),
            Vector3(get_random_value(1, CUBE_MAX), get_random_value(1, CUBE_MAX), get_random_value(1, CUBE_MAX) ),
        ))


camera = Camera3D(
    Vector3(30, 70,-25),
    Vector3(0,0,-25),
    Vector3(0,1,0),
    60.0,
    CAMERA_PERSPECTIVE
)


init_window(800, 450, "Hello")
set_target_fps(60)

frame_index = 0

while not window_should_close():
    
    frameTime = get_frame_time()

    for c in cubes:
        c.position = vector3_add(c.position, vector3_scale(c.velocity, frameTime))
    
    begin_drawing()
    begin_mode_3d(camera)
    clear_background(WHITE)
    
    for c in cubes:
        center = vector3_scale( vector3_add(c.BoundingBox_min, c.BoundingBox_max), 0.5)
        size = vector3_subtract(c.BoundingBox_min, c.BoundingBox_max)
        draw_cube(
            vector3_add(c.position, center),
            size.x, size.y, size.z,
            c.color
        )

    frame_index+=1
    if frame_index%10 == 0 and get_fps() > 50: add_cubes()
    
    end_mode_3d()
    draw_text(f"fps {get_fps()} cubes {len(cubes)} ", 10, 10, 20, LIGHTGRAY)
    end_drawing()
close_window()

