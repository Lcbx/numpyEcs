from raylib import *
from pyray import Vector2, Vector3, Color, Camera3D, rl_load_texture, RenderTexture
from ecs import *

#import raylib as rl
#print(dir(rl))
#exit(0)

@component
class Position:
    x: float; y: float; z: float

@component
class Velocity:
    x: float; y: float; z: float

@component
class Mesh:
    color : Color
    
@component
class BoundingBox:
    x_min: float; y_min: float; z_min: float
    x_max: float; y_max: float; z_max: float

world = ECS()
world.register(Position, Velocity, Mesh, BoundingBox)

positions = world.get_store(Position)
velocities = world.get_store(Velocity)
meshes = world.get_store(Mesh)
bboxes = world.get_store(BoundingBox)

ground = world.create_entity(
    Position(0,-0.51,0),
    Mesh(LIGHTGRAY),
    BoundingBox(
        -25,0,-25,
        25,1,25
    )
)

rnd_uint8 = lambda : GetRandomValue(0, 255)
rnd_color = lambda : Color(rnd_uint8(),rnd_uint8(),rnd_uint8(),255)

SPACE_SIZE = 30
CUBE_MAX = 7

for e in world.create_entities(15):
    world.add_component(e,
        Position(
            GetRandomValue(-SPACE_SIZE, SPACE_SIZE),
            GetRandomValue(0, 25),
            GetRandomValue(-SPACE_SIZE, SPACE_SIZE) ),
        Velocity(
            GetRandomValue(-4, 4),
            0,
            GetRandomValue(-4, 4) ),
        BoundingBox(
            GetRandomValue(-CUBE_MAX, 0), GetRandomValue(-CUBE_MAX, 0), GetRandomValue(-CUBE_MAX, 0),
            GetRandomValue(1, CUBE_MAX), GetRandomValue(1, CUBE_MAX), GetRandomValue(1, CUBE_MAX) ),
        Mesh(rnd_color()),
    )


camera = Camera3D(
    Vector3(30, 70,-25),
    Vector3(0,0,-25),
    Vector3(0,1,0),
    60.0,
    CAMERA_PERSPECTIVE
)

light_camera = Camera3D(
    Vector3(-20, 30, 5),
    Vector3(0,0,0),
    Vector3(0,1,0),
    90.0,
    CAMERA_ORTHOGRAPHIC
)

WINDOW_SIZE = Vector2(800, 450) 
InitWindow(int(WINDOW_SIZE.x), int(WINDOW_SIZE.y), b"Hello")
SetTargetFPS(60)


def load_shaders():
    global sceneShader
    newShader = LoadShader(b"scenes/lightmap.vs", b"scenes/lightmap.fs")
    if newShader.id > 0: sceneShader = newShader

sceneShader = None
load_shaders()

def shadow_buffer(width : int, height:int, withColorBuffer:bool=False) -> RenderTexture :
    # has a color buffer by default
    #target = LoadRenderTexture(width, height)

    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    
    if target.id > 0:
        rlEnableFramebuffer(target.id)
        
        if withColorBuffer:
            target.texture.id = rl_load_texture(None, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
            target.texture.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
            target.texture.mipmaps = 1
            rlFramebufferAttach(target.id, target.texture.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)

        target.texture.width = width
        target.texture.height = height

        target.depth.id = rlLoadTextureDepth(width, height, False)
        target.depth.width = width
        target.depth.height = height
        target.depth.format = 19 # ?
        target.depth.mipmaps = 1

        rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
        rlDisableFramebuffer()

    return target

shadowmap = shadow_buffer(1024,1024,withColorBuffer=True)


def run():
    while not WindowShouldClose():
        
        frameTime = GetFrameTime()

        if IsKeyPressed(KEY_R): load_shaders()

        pv = world.where(Position, Velocity)
        p_vec, v_vec = (positions.get_vector(pv), velocities.get_vector(pv))
        p_vec += v_vec * frameTime

        # bounce when out of bounds
        mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
        mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
        v_vec[mask_x, 0] *= -1
        v_vec[mask_z, 2] *= -1

        positions.set_vector(pv, p_vec)
        velocities.set_vector(pv, v_vec)
        
        BeginDrawing()
        BeginMode3D(camera)
        ClearBackground(WHITE)
        
        draw_scene()
        
        EndMode3D()
        draw_shadowmap()
        DrawText(f"fps {GetFPS()} cubes {world.count} ".encode('utf-8'), 10, 10, 20, LIGHTGRAY)
        EndDrawing()
    CloseWindow()


def draw_shadowmap():
    display_size = WINDOW_SIZE.x / 5.0
    display_scale = display_size / float(shadowmap.depth.width)
    DrawTextureEx(shadowmap.texture, Vector2(WINDOW_SIZE.x - display_size, 0.0), 0.0, display_scale, RAYWHITE)
    DrawTextureEx(shadowmap.depth, Vector2(WINDOW_SIZE.x - display_size, display_size), 0.0, display_scale, RAYWHITE)

def draw_scene():
    ents = world.where(Position, Mesh, BoundingBox)
    pos_vec, mesh_vec, bb_vec, = (positions.get_vector(ents), meshes.get_vector(ents), bboxes.get_vector(ents))
    bmins = bb_vec[:,:3] # entity (int), bounding box (6 floats)
    bmaxs = bb_vec[:,3:]
    sizes = bmaxs - bmins
    centers = pos_vec + (bmaxs + bmins) * 0.5

    for center, size, mesh in zip(centers, sizes, mesh_vec):
        DrawCube(
            tuple(center),
            size[0], # x
            size[1], # y
            size[2], # z
            mesh[meshes.color_id]
        )


run()