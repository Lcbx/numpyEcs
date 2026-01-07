import math
import time
import numpy as np

import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop

from pyrr import Matrix44, Vector3


# ----------------------------
# Config
# ----------------------------
WINDOW_W, WINDOW_H = 1800, 900
TITLE = "pyGfx + GLSL (experimental) + glTF"
MODEL_PATH = "scenes/resources/rooftop_utility_pole.glb"

canvas = RenderCanvas(
    size=(WINDOW_W, WINDOW_H), update_mode="fastest", title=TITLE, vsync=False
)

renderer = gfx.WgpuRenderer(canvas, ppaa='none', pixel_filter='nearest')

camera = gfx.PerspectiveCamera(45, WINDOW_W / WINDOW_H)
#camera.local.position = (-20.0, 70.0, 25.0)

def setMaterial(obj):
     if isinstance(obj, gfx.Mesh) and isinstance(obj.material, gfx.MeshStandardMaterial):
        obj.material = gfx.MeshPhongMaterial(color="#d1b58c")

gltf = gfx.load_gltf(MODEL_PATH)
model = gltf.scene if gltf.scene else gltf.scenes[0]
model.traverse(setMaterial)
model.local.scale = 5.0

scene = gfx.Scene()
scene.add(gfx.Background.from_color((0.15, 0.16, 0.19, 1.0)))
scene.add(gfx.AmbientLight(1.5))
scene.add(model)

camera.show_object(scene)
camera.target = model.local
#controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    global fps_frames, fps_t0

    now = time.time()
    elapsed = now - start_t

    cam_ang = elapsed * 0.5
    camera.local.x = math.cos(cam_ang) * camera_dist
    camera.local.y = 70
    camera.local.z = math.sin(cam_ang) * camera_dist
    camera.look_at(model.children[0].local.position)

    # FPS
    fps_frames += 1
    if now - fps_t0 >= 1.0:
        print(f"fps {fps_frames}")
        fps_t0 = now
        fps_frames = 0

    renderer.render(scene, camera)
    renderer.request_draw()

camera_dist = 30.0
start_t = time.time()
fps_t0 = time.time()
fps_frames = 0

renderer.request_draw(animate)
loop.run()