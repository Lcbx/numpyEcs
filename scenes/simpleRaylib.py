import numpy as np
import time
import shader_util as su
from shader_util import Vector2, Vector3, Color, Camera3D, RenderTexture


WINDOW_w, WINDOW_h = 1800, 900
su.InitWindow(WINDOW_w, WINDOW_h, "OpenGL 3.3 Window")


model_root = b'scenes/resources/'
model = su.rl.LoadModel(model_root + b'rooftop_utility_pole.glb')

camera_dist = 30
camera_nearFar = (0.1, 1000.0)
camera = Camera3D(
    Vector3(-20, 70, 25),
    Vector3(0,10,0),
    Vector3(0,1,0),
    60.0,
    su.rl.CAMERA_PERSPECTIVE
)


lightDir = su.rl.Vector3Normalize(su.rl.Vector3Subtract(Vector3(30, 30, 25), Vector3(0,0,-20)))

def load_shaders():
    global sceneShader

    # Bettersahder is a utility class built on top of raylib shader compilation
    # no need to convert it
    newShader = su.BetterShader('scenes/minimalShader.shader')
    if newShader.valid(): sceneShader = newShader
    else: raise Exception('minimalShader.shader')


sceneShader : su.BetterShader = None
load_shaders()

def run():
    start_t = time.time()
    frame_start = 0.0
    fps_frames = 0

    while not su.rl.WindowShouldClose():
        inputs()

        # per-frame timing
        now = time.time()
        elapsed = now - start_t

        update(elapsed)

        su.rl.BeginDrawing()
        
        with su.RenderContext(shader=sceneShader, clipPlanes=camera_nearFar, camera=camera) as render:
        
            su.ClearBuffers()
            
            sceneShader.lightDir = lightDir
            draw_scene(render)

        su.rl.EndDrawing()

        # simple FPS to console
        fps_frames += 1
        if now - frame_start >= 1.0:
            print(f"fps {fps_frames}")
            frame_start = time.time()
            fps_frames = 0


orbit = True
def inputs():
    global camera
    global camera_dist
    global orbit

    if su.rl.IsKeyPressed(su.rl.KEY_R): load_shaders()
    if su.rl.IsKeyPressed(su.rl.KEY_O): orbit = not orbit

    if su.rl.IsKeyPressed(su.rl.KEY_P):
        light_camera.projection = (su.rl.CAMERA_ORTHOGRAPHIC
            if light_camera.projection == su.rl.CAMERA_PERSPECTIVE
            else su.rl.CAMERA_PERSPECTIVE
        )

    scrollspeed = 3.0
    mw = scrollspeed * su.rl.GetMouseWheelMove()
    if mw != 0 and ( camera.position.y > scrollspeed + 0.5 or mw > 0.0):
        cam_pos = np.array([camera.position.x, camera.position.y, camera.position.z])
        tar = np.array([camera.target.x, camera.target.y, camera.target.z])
        camera_dist -= mw * 0.3;
        cam_pos[1] += (tar[1] - cam_pos[1]) / np.linalg.norm(cam_pos[1] + 0.1) * mw
        camera.position = Vector3(cam_pos[0], cam_pos[1], cam_pos[2])


def update(elapsed):
    global camera
    
    if orbit:
        cam_ang = elapsed * 0.5
        camera.position = Vector3(
            np.cos(cam_ang) * camera_dist,
            camera.position.y,
            np.sin(cam_ang) * camera_dist)


def draw_scene(render:su.RenderContext, randomize_color=False):

    for i in range(model.materialCount):
        model.materials[i].shader = render.shader.shaderStruct

    scale = 5.0
    su.rl.DrawModelEx(model, Vector3(0,0,0), Vector3(1,0,0), 0.0, Vector3(scale,scale,scale), su.rl.BEIGE)


run()