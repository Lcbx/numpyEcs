from RenderContext import *
from Utils import *
from ECS import *
from math import cos, sin
import random as rd

# maybe collapse the whole thing into a movingMesh component ?
# or just use instance_dtype directly in the ecs ?

@component
class Position:
	x: float; y: float; z: float

@component
class Velocity:
	x: float; y: float; z: float

@component
class Rotation:
	x: float; y: float; z: float; w:float

@component
class Scale:
	x: float; y: float; z: float; w:float # w is not used

#@component
#class MeshInstance:
#	data : mesh_instance_dtype

@component
class Tint:
	value: np.uint32


world = ECS()
world.register(Position, Velocity, Rotation, Scale, Tint)

positions = world.get_store(Position)
velocities = world.get_store(Velocity)
rotations = world.get_store(Rotation)
scales = world.get_store(Scale)
tints = world.get_store(Tint)

SPACE_SIZE = 180
CUBE_MAX_SIDE = 7

ground = world.create_entity(
	Position(0,-0.51,0),
	Rotation( *Quaternion() ),
	Scale(
		2*SPACE_SIZE, 1, 2*SPACE_SIZE,
	0),
	Tint(
		pack_rgba8_srgb([0.5, 0.5, 0.5, 1.0])
	),
)

for e in world.create_entities(200):
	world.add_component(e,
		Position(
			rd.randrange(-SPACE_SIZE, SPACE_SIZE),
			rd.randrange(0, 20),
			rd.randrange(-SPACE_SIZE, SPACE_SIZE)
		),
		Velocity(
			rd.randrange(-4, 4),
			0,
			rd.randrange(-4, 4)
		),
		Rotation( *Quaternion() ),
		Scale(
			rd.randrange(1, CUBE_MAX_SIDE),
			rd.randrange(1, CUBE_MAX_SIDE),
			rd.randrange(1, CUBE_MAX_SIDE),
			0),
		Tint(
			pack_rgba8_srgb([rd.random(), rd.random(), rd.random(), 1.0])
		),
	)



WINDOW_W, WINDOW_H = 1200, 1200
TITLE = "glfw + wgpu"


camera_dist = 30.0
camera = Camera(
	position=(-20.0, 70.0, 25.0),
	target=(0.0, 10.0, 0.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=60.0,
	near=0.1, far=1000.0
)

light_camera = Camera(
	position=(-30.0, 30.0, 25.0),
	target=(0.0, 0.0, -20.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=90.0,
	near=1.0, far=300.0,
	perspective = False
)


RenderContext.InitWindow(WINDOW_W, WINDOW_H, TITLE
#)
#, target_fps=120)
, target_fps=-1)
#RenderContext.capture_mouse()

# seems to use a weird color space
renderpass = RenderContext.RenderPass(camera = camera, clear_color = (0.02, 0.02, 0.03, 1.0))
shader = RenderContext.Shader(filepath='scenes/shaders/simple.shader')
shadow_shader = RenderContext.Shader(filepath='scenes/shaders/prepass.shader')
vertices, indices = load_gltf_first_mesh_interleaved('scenes/resources/rooftop_utility_pole.glb')
scale = 10.0
model_mesh = Mesh(vertices, indices)
model_instances = np.zeros(2, mesh_instance_dtype) 
model_instances[0]["iPosition"] = Vec3([15.0, 0.0, 15.0])
model_instances[0]["iRotation"] = Quaternion()
model_instances[0]["iScale"] = [scale] * 4 # only 3 are used
model_instances[0]["iTint"] = pack_rgba8_srgb(Vec4([0.3, 0.5, 0.7, 1.0]))
model_mesh.set_instances(model_instances)

cube_mesh = RenderContext.resources["cube"]


uniformBuffer = shader.UniformBuffer()

shadow_texture = create_depth_framebuffer(512, 512)
shadow_view = shadow_texture.create_view()
shadow_rp = RenderContext.RenderPass(camera = light_camera,
	frame_buffers = (None, shadow_view)
)
shadow_sampler = create_depth_sampler()


print('init done')

def clamp(val, val_min, val_max):
	return min(max(val, val_min), val_max)

def scroll_callback(xoff, yoff):
	global camera_dist, camera
	camera_dist = clamp(camera_dist - 5.0 * yoff, 5.0, 100.0)
	cp = camera.position
	y_factor = 1.0 if cp.y < 15.0 else 3.0
	#print(f'{y_factor=}')
	camera.position = Vec3( (cp.x, cp.y - y_factor * yoff, cp.z) )
	return True

RenderContext.event_handlers['mouse_scroll'].append(scroll_callback)

#setup_gc_monitor()
#print_memory = setup_memory_monitor()


fps_frames = 0
start_t = getTime()
fps_print_timestamp = start_t
while RenderContext.WindowLoop():
	now = RenderContext.frame_start

	# simple FPS to console
	fps_frames += 1
	if now - fps_print_timestamp >= 1.0:
		print(f"fps {fps_frames}")
		fps_frames = 0
		fps_print_timestamp = now
		
		model_instances[1]["iPosition"] = Vec3([np.random.rand()*15.0, 0.0, np.random.rand()*15.0])
		model_instances[1]["iRotation"] = Quaternion()
		model_instances[1]["iScale"] = [scale] * 4 # only 3 are used
		model_instances[1]["iTint"] = pack_rgba8_srgb(Vec4([0.6, 0.5, 0.4, 1.0]))

		model_mesh.instance_buffer.upload()

		#camera.perspective = not camera.perspective

		#print(WatchTimer.capture())
		#print_memory()
		#print("")


	# orbit update (based on time wince start)
	elapsed = now - start_t
	cam_ang = elapsed * 0.5
	camera.position = Vec3( (
		cos(cam_ang) * camera_dist,
		camera.position.y,
		sin(cam_ang) * camera_dist
	) )


	# cubes movement

	pv = world.where(Position, Velocity)
	p_vec, v_vec = (positions.get_full_vector(pv), velocities.get_full_vector(pv))
	p_vec += v_vec * RenderContext.frame_time

	# bounce when out of bounds
	mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
	mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
	v_vec[mask_x, 0] *= -1
	v_vec[mask_z, 2] *= -1
	# if further than boundary, it would get stuck alternating direction each frame
	p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
	p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE

	positions.set_full_vector(pv, p_vec.transpose())
	velocities.set_full_vector(pv, v_vec.transpose())

	cube_mesh.set_instances( make_instances(
		positions.get_vector(),
		rotations.get_vector(),
		scales.get_vector(),
		tints.get_vector()
	))

	with WatchTimer("draw"):

		with renderpass as rp:

			# shadows, from light viewpoint
			with shadow_rp as rp2:

				with WatchTimer("upload_buffers"):
					cube_mesh.instance_buffer.upload()

					uniformBuffer.content['uView'] = renderpass.view
					uniformBuffer.content['uProj'] = renderpass.projection
					uniformBuffer.content['uInverseProj'] = np.linalg.inv(renderpass.projection)
					uniformBuffer.content['uLightDir'] = light_camera.direction()
					uniformBuffer.content['uLightViewProj'] = shadow_rp.view @ shadow_rp.projection
					uniformBuffer.upload()

				with WatchTimer("draw_model"):
					model_mesh.draw(rp2, shadow_shader, uniformBuffer)
					model_mesh.draw(rp, shader, uniformBuffer, textureSamplerGroups = [(shadow_view,shadow_sampler)])

				with WatchTimer("draw_cubes"):
					cube_mesh.draw(rp2, shadow_shader, uniformBuffer)
					cube_mesh.draw(rp, shader, uniformBuffer, textureSamplerGroups = [(shadow_view,shadow_sampler)])

			# TODO : use shadow map results in regular renderpass

