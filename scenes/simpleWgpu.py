import math
import time
import numpy as np

import shader_util as su

import wgpu
from rendercanvas.auto import RenderCanvas, loop

from pyrr import Matrix44, Vector3


# ----------------------------
# Config
# ----------------------------
WINDOW_W, WINDOW_H = 1800, 900
TITLE = "wgpu-py + GLSL (experimental) + glTF"
MODEL_PATH = "scenes/resources/rooftop_utility_pole.glb"


# ----------------------------
# GLSL (wgpu binding model: set/binding)
# ----------------------------
VERT_SRC = r"""
#version 450

layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

layout(set=0, binding=0, std140) uniform Uniforms {
    mat4 uModel;
    mat4 uView;
    mat4 uProj;
    vec3 uLightDir;
} U;

layout(location=0) out vec3 vNormalWS;
layout(location=1) out vec3 vLightDirWS;
layout(location=2) out vec2 vUV;

void main() {
    vec4 worldPos = U.uModel * vec4(aPos, 1.0);
    vNormalWS = mat3(U.uModel) * aNormal; // matches your original (no inv-transpose)
    vLightDirWS = normalize(U.uLightDir);
    vUV = aUV;
    gl_Position = U.uProj * U.uView * worldPos;
}
"""

FRAG_SRC = r"""
#version 450

layout(location=0) in vec3 vNormalWS;
layout(location=1) in vec3 vLightDirWS;
layout(location=2) in vec2 vUV;

layout(location=0) out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormalWS);
    float NdotL = max(dot(N, vLightDirWS), 0.0);
    vec3 base = vec3(0.82, 0.71, 0.55);
    vec3 color = base * (0.2 + 0.8 * NdotL);
    FragColor = vec4(color, 1.0);
}
"""


def load_gltf_first_mesh_interleaved(glb_path: str):
    ( pos, nor, uv, idx ) = su.load_gltf_first_mesh(glb_path)
    # interleave: pos.xyz, normal.xyz, uv.xy => 8 floats => 32 bytes
    v = np.concatenate([pos, nor, uv], axis=1).astype(np.float32)  # (N, 8)
    return v, idx.reshape(-1).astype(np.uint32)


def main():
    # --- Canvas / context / device
    canvas = RenderCanvas(
        title=TITLE,
        size=(WINDOW_W, WINDOW_H),
        update_mode="continuous",
        max_fps=100000,
        vsync=False,
        #max_fps=60,
        #vsync=True,
    )
    context = canvas.get_wgpu_context()

    adapter = wgpu.gpu.request_adapter_sync(canvas=canvas, power_preference="high-performance")
    device = adapter.request_device_sync(required_limits={})

    presentation_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=presentation_format, usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

    # --- App state
    camera_dist = 30.0
    camera_pos = Vector3((-20.0, 70.0, 25.0))
    camera_target = Vector3((0.0, 10.0, 0.0))
    camera_up = Vector3((0.0, 1.0, 0.0))
    fovy_deg = 60.0
    near, far = 0.1, 1000.0
    orbit = True

    start_t = time.time()
    fps_t0 = time.time()
    fps_frames = 0

    # --- Model + light
    scale = 5.0
    model_mat = Matrix44.from_scale([scale, scale, scale], dtype=np.float32)

    l = np.array([30.0, 30.0, 25.0], dtype=np.float32) - np.array([0.0, 0.0, -20.0], dtype=np.float32)
    l = l / (np.linalg.norm(l) + 1e-8)
    light_dir = l.astype(np.float32)

    # --- Mesh buffers
    vertices, indices = load_gltf_first_mesh_interleaved(MODEL_PATH)
    index_count = int(indices.size)

    vertex_buffer = device.create_buffer_with_data(data=vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX)
    index_buffer = device.create_buffer_with_data(data=indices.tobytes(), usage=wgpu.BufferUsage.INDEX)

    # --- Uniform buffer (std140): 3*mat4 (64 each) + vec3 + pad = 208
    uniform_buffer_size = 64 * 3 + 18
    uniform_buffer = device.create_buffer(
        size=uniform_buffer_size,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    bind_group_layout = device.create_bind_group_layout(entries=[
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }
    ])
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[{
            "binding": 0,
            "resource": {"buffer": uniform_buffer, "offset": 0, "size": uniform_buffer_size},
        }],
    )
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

    # --- GLSL shader modules (wgpu-py GLSL is “experimental”)
    # Label must contain "vert"/"frag" so wgpu-py can infer the stage (as in their GLSL example).
    vert_module = device.create_shader_module(label="shader.vert", code=VERT_SRC)
    frag_module = device.create_shader_module(label="shader.frag", code=FRAG_SRC)

    # --- Depth texture helper
    depth_format = wgpu.TextureFormat.depth24plus
    depth_tex = None
    depth_view = None

    w, h = canvas.get_logical_size()
    w = max(1, int(w))
    h = max(1, int(h))
    depth_tex = device.create_texture(
        size=(w, h, 1),
        format=depth_format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )
    depth_view = depth_tex.create_view()

    # --- Pipeline
    pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": vert_module,
            "entry_point": "main",
            "buffers": [{
                "array_stride": 8 * 4,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {"shader_location": 0, "offset": 0 * 4, "format": wgpu.VertexFormat.float32x3},
                    {"shader_location": 1, "offset": 3 * 4, "format": wgpu.VertexFormat.float32x3},
                    {"shader_location": 2, "offset": 6 * 4, "format": wgpu.VertexFormat.float32x2},
                ],
            }],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.back,
            "front_face": wgpu.FrontFace.ccw,
        },
        depth_stencil={
            "format": depth_format,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        },
        fragment={
            "module": frag_module,
            "entry_point": "main",
            "targets": [{"format": presentation_format}],
        },
    )

    # --- Events (use add_event_handler per docs) :contentReference[oaicite:2]{index=2}
    def handle_event(event):
        nonlocal orbit, camera_dist, camera_pos
        #print(event)

        et = event.get("event_type", "")

        if et == "key_down":
            if event.get("key", "").lower() == "o":
                orbit = not orbit

        elif et == "wheel":
            dy = float(event.get("dy", 0.0))
            scrollspeed = 0.1
            mw = scrollspeed * dy
            if mw != 0 and (camera_pos.y > scrollspeed + 0.5 or mw > 0.0):
                tar = np.array([camera_target.x, camera_target.y, camera_target.z], dtype=np.float32)
                cam = np.array([camera_pos.x, camera_pos.y, camera_pos.z], dtype=np.float32)
                camera_dist -= mw * 0.3
                cam[1] += (tar[1] - cam[1]) / (abs(cam[1]) + 0.1) * mw
                camera_pos = Vector3(cam)

    canvas.add_event_handler(handle_event, '*') #'before_draw')

    # --- Draw
    def draw_frame():
        nonlocal camera_pos, fps_frames, fps_t0

        now = time.time()
        elapsed = now - start_t

        if orbit:
            cam_ang = elapsed * 0.5
            camera_pos = Vector3((
                math.cos(cam_ang) * camera_dist,
                camera_pos.y,
                math.sin(cam_ang) * camera_dist
            ))

        w, h = canvas.get_logical_size()
        aspect = float(w) / max(1.0, float(h))

        view = Matrix44.look_at(camera_pos, camera_target, camera_up, dtype=np.float32)
        proj = Matrix44.perspective_projection(fovy_deg, aspect, near, far, dtype=np.float32)

        # std140 pack: model + view + proj + (light vec3 + pad)
        light4 = np.zeros(4, dtype=np.float32)
        light4[:3] = light_dir

        blob = (
            np.asarray(model_mat, dtype=np.float32).tobytes() +
            np.asarray(view, dtype=np.float32).tobytes() +
            np.asarray(proj, dtype=np.float32).tobytes() +
            light4.tobytes()
        )
        device.queue.write_buffer(uniform_buffer, 0, blob)

        current_texture = context.get_current_texture()

        color_view = current_texture.create_view()

        encoder = device.create_command_encoder()
        rp = encoder.begin_render_pass(
            color_attachments=[{
                "view": color_view,
                "resolve_target": None,
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.15, 0.16, 0.19, 1.0), # for some reason it is not the right shade
            }],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )

        rp.set_pipeline(pipeline)
        rp.set_bind_group(0, bind_group, [])
        rp.set_vertex_buffer(0, vertex_buffer)
        rp.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)
        rp.draw_indexed(index_count, 1, 0, 0, 0)
        rp.end()

        device.queue.submit([encoder.finish()])

        # FPS
        fps_frames += 1
        if now - fps_t0 >= 1.0:
            print(f"fps {fps_frames}")
            fps_t0 = now
            fps_frames = 0

    canvas.request_draw(draw_frame)

    loop.run()


main()
