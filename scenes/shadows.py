from ecs import *
import shader_util as su
from shader_util import Vector2, Vector3, Color, Camera3D, RenderTexture

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

rnd_uint8 = lambda : su.rl.GetRandomValue(0, 255)
rnd_color = lambda : Color(rnd_uint8(),rnd_uint8(),rnd_uint8(),255)

SPACE_SIZE = 180
CUBE_MAX_SIDE = 7

for e in world.create_entities(200):
	world.add_component(e,
		Position(
			su.rl.GetRandomValue(-SPACE_SIZE, SPACE_SIZE),
			su.rl.GetRandomValue(0, 20),
			su.rl.GetRandomValue(-SPACE_SIZE, SPACE_SIZE) ),
		Velocity(
			su.rl.GetRandomValue(-4, 4),
			0,
			su.rl.GetRandomValue(-4, 4) ),
		BoundingBox(
			su.rl.GetRandomValue(-CUBE_MAX_SIDE, 0), su.rl.GetRandomValue(-CUBE_MAX_SIDE, 0), su.rl.GetRandomValue(-CUBE_MAX_SIDE, 0),
			su.rl.GetRandomValue(1, CUBE_MAX_SIDE),  su.rl.GetRandomValue(1, CUBE_MAX_SIDE),  su.rl.GetRandomValue(1, CUBE_MAX_SIDE) ),
		Mesh(rnd_color()),
	)


def load_shaders():
	global shadowMeshShader
	global shadowBlurShader
	global sceneShader
	global prepassShader
	global AOshader
	global kawaseBlur_downSampleShader
	global kawaseBlur_upSampleShader

	try:
		newShader = su.BetterShader('scenes/shadowmesh.shader');
		if newShader.valid(): shadowMeshShader = newShader
		else: raise Exception('shadowmesh.shader')

		newShader = su.BetterShader('scenes/lightmap.compute');
		if newShader.valid(): shadowBlurShader = newShader
		else: raise Exception('lightmap.compute')

		newShader = su.BetterShader('scenes/lightmap.shader')
		if newShader.valid(): sceneShader = newShader
		else: raise Exception('lightmap.shader')

		newShader = su.BetterShader('scenes/prepass.shader')
		if newShader.valid(): prepassShader = newShader
		else: raise Exception('prepass.shader')
		
		newShader = su.BetterShader('scenes/AO.compute')
		if newShader.valid(): AOshader = newShader
		else: raise Exception('AO.compute')
		
		newShader = su.BetterShader('scenes/kawaseBlur_downSample.compute')
		if newShader.valid(): kawaseBlur_downSampleShader = newShader
		else: raise Exception('kawaseBlur_downSample.compute')

		newShader = su.BetterShader('scenes/kawaseBlur_upSample.compute')
		if newShader.valid(): kawaseBlur_upSampleShader = newShader
		else: raise Exception('kawaseBlur_upSampleShader.compute')

	except Exception as ex:
		print('--------------> failed compiling', ex.args[0])
		print(newShader.vertex_glsl)
		print('______________________')
		print(newShader.fragment_glsl)

#WINDOW_w, WINDOW_h = 1000, 650
#WINDOW_w, WINDOW_h = 1920, 1080
WINDOW_w, WINDOW_h = 1800, 900
su.InitWindow(WINDOW_w, WINDOW_h, "Hello")


sceneShader : su.BetterShader = None
shadowMeshShader : su.BetterShader = None
shadowBlurShader : su.BetterShader = None
prepassShader : su.BetterShader = None
AOshader : su.BetterShader = None
depthBlurShader : su.BetterShader = None
kawaseBlur_downSampleShader : su.BetterShader = None
kawaseBlur_upSampleShader : su.BetterShader = None
load_shaders()

camera_dist = 30
camera_nearFar = (0.1, 1000.0)
camera = Camera3D(
	Vector3(-20, 70, 25),
	Vector3(0,10,0),
	Vector3(0,1,0),
	60.0,
	su.rl.CAMERA_PERSPECTIVE
)

light_nearFar = (10,300)
light_camera = Camera3D(
	Vector3(30, 30, 25),
	Vector3(0,0,-20),
	Vector3(0,1,0),
	90.0,
	su.rl.CAMERA_ORTHOGRAPHIC
)

unused_camera = None

# TODO : handle resolution changes (rebuild buffers)
# None is for color format, means we dont actually draw into it
prepass_buffer = su.create_render_buffer(WINDOW_w, WINDOW_h, None, depth_map=True)
AO_w, AO_h = WINDOW_w//2, WINDOW_h//2
AO_buffer = su.create_render_buffer(AO_w, AO_h, su.rl.PIXELFORMAT_UNCOMPRESSED_R16)
AO_buffer2 = su.create_render_buffer(AO_w//2, AO_h//2, su.rl.PIXELFORMAT_UNCOMPRESSED_R16)
AO_buffer3 = su.create_render_buffer(AO_w//4, AO_h//4, su.rl.PIXELFORMAT_UNCOMPRESSED_R16)

SM_SIZE = 1024
INV_SM_SIZE = 1.0 / float(SM_SIZE)
SHADOW_FORMAT = su.rl.PIXELFORMAT_UNCOMPRESSED_R16G16B16
shadow_buffer = su.create_render_buffer(SM_SIZE,SM_SIZE,colorFormat=SHADOW_FORMAT, depth_map=True)
shadow_buffer2 = su.create_render_buffer(SM_SIZE,SM_SIZE,colorFormat=SHADOW_FORMAT)


# model
model_root = b'scenes/resources/'
#model = su.rl.LoadModel(model_root + b'teapot.obj')
model = su.rl.LoadModel(model_root + b'turret.obj')
model_albedo = su.rl.LoadTexture(model_root + b'turret_diffuse.png')
su.SetMaterialTexture(model.materials[0], su.rl.MATERIAL_MAP_DIFFUSE, model_albedo)
heightmap = su.rl.LoadModel(model_root + b'heightmap_mesh.glb')

#anims = su.LoadModelAnimations(model_root + b'mixamo_toon_gisu.rl.glb')
#animFrameCounter = 0

def run():
	while not su.rl.WindowShouldClose():

		frameTime = su.rl.GetFrameTime()
		
		with su.WatchTimer('update'):
			inputs()
			update(frameTime)
		
		with su.WatchTimer('total draw'):
			
			with su.WatchTimer('shadow'):

				with su.RenderContext(shader=shadowMeshShader, texture=shadow_buffer, camera=light_camera, clipPlanes=light_nearFar) as render:
					su.ClearBuffers()
					su.SetPolygonOffset(3.0) # should increase to 3 for perspective light
					
					lightDir = su.rl.Vector3Normalize(su.rl.Vector3Subtract(light_camera.position, light_camera.target))
					lightVP = su.rl.MatrixMultiply(su.rl.rlGetMatrixModelview(), su.rl.rlGetMatrixProjection())

					draw_scene(render,randomize_color=True)
					
					su.DisablePolygonOffset()

				# populate fuzzy shadow map with passes
				read_buffer = shadow_buffer
				write_buffer = shadow_buffer2
				step = 8.0
				last = 1.0
				
				#shadowBlurShader.depthmap = shadow_buffer.depth
				while step > last:
					with su.RenderContext(shader=shadowBlurShader, texture=write_buffer) as render:
						step *= 0.5
						shadowBlurShader.stepSize = step * INV_SM_SIZE
						#shadowBlurShader.last = 1 if step <= last else 0
						su.DrawTexture(read_buffer.texture, SM_SIZE, SM_SIZE)
						read_buffer, write_buffer = write_buffer, read_buffer

			# main camera
			with su.WatchTimer('main camera'):
				# TODO : don't do frustum culling twice (prepass + main pass)
				
				# z prepass
				with su.WatchTimer('prepass'):
					with su.RenderContext(shader=prepassShader, texture=prepass_buffer, clipPlanes=camera_nearFar, camera=camera) as render:
						proj = su.rl.rlGetMatrixProjection()
						su.ClearBuffers()
						su.SetPolygonOffset(0.1)
						draw_scene(render)
						su.DisablePolygonOffset()
				
				# AO
				with su.WatchTimer('AO'):
					if applyAO:
						# TODO : maybe generate mipmaps earlier and use them at other places ?
						su.GenTextureMipmaps(prepass_buffer.depth)

						with su.RenderContext(shader=AOshader, texture=AO_buffer) as render:
							AOshader.invProj = su.rl.MatrixInvert(proj)
							su.DrawTexture(prepass_buffer.depth, AO_w, AO_h)

						# kawase-blur : good perf no matter the kernel size
						# + easy choose a compromise between quality ands cost
						# link: https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf

						pixel_scale = Vector2(1.0/float(AO_w),1.0/float(AO_h))
						pixel_scale_x2 = su.rl.Vector2Scale(pixel_scale, 2)
						pixel_scale_x4 = su.rl.Vector2Scale(pixel_scale_x2, 2)

						with su.RenderContext(shader=kawaseBlur_downSampleShader, texture=AO_buffer2) as render:
							kawaseBlur_downSampleShader.u_direction = pixel_scale
							su.DrawTexture(AO_buffer.texture, AO_w, AO_h)

						with su.RenderContext(shader=kawaseBlur_downSampleShader, texture=AO_buffer3) as render:
							kawaseBlur_downSampleShader.u_direction = pixel_scale_x2
							su.DrawTexture(AO_buffer2.texture, AO_w, AO_h)

						with su.RenderContext(shader=kawaseBlur_upSampleShader, texture=AO_buffer2) as render:
							kawaseBlur_upSampleShader.u_direction = pixel_scale_x4
							su.DrawTexture(AO_buffer3.texture, AO_w, AO_h)

						with su.RenderContext(shader=kawaseBlur_upSampleShader, texture=AO_buffer) as render:
							kawaseBlur_upSampleShader.u_direction = pixel_scale_x2
							su.DrawTexture(AO_buffer2.texture, AO_w, AO_h)
				
				with su.WatchTimer('forward pass'):

					su.rl.BeginDrawing()
					
					# transfer depth to main buffer for early z discard
					su.TransferDepth(prepass_buffer.id, WINDOW_w, WINDOW_h, 0, WINDOW_w, WINDOW_h)

					with su.RenderContext(shader=sceneShader, clipPlanes=camera_nearFar, camera=camera) as render:
					
						su.ClearColorBuffer() # do not clear depth !
						
						sceneShader.lightDir = lightDir
						sceneShader.lightVP  = lightVP
						sceneShader.shadowSamplingRadius = 3.5 * INV_SM_SIZE
						sceneShader.shadowDepthMap = shadow_buffer.depth
						sceneShader.shadowPenumbraMap = read_buffer.texture
						sceneShader.ambientOcclusionMap = AO_buffer.texture
						draw_scene(render)
				
				# maybe add toggles for drawing buffers
				#draw_shadow_buffer()
				#draw_prepass()
				#draw_AO()
		
			su.rl.DrawText(f"fps {su.rl.GetFPS()} cubes {world.count}".encode(), 10, 10, 20, su.rl.LIGHTGRAY)
			su.WatchTimer.display(10, 40, 20, su.rl.LIGHTGRAY)

			# sleeps for vsync / target fps
			su.rl.EndDrawing()

			# a lot of stuff happening ain EndDrawing it seems...
			# shadow map cost seems to determine perf the most
			su.WatchTimer.capture()


rotation = 0
def draw_shadow_buffer():
	display_size = WINDOW_w / 5.0
	display_scale = display_size / float(shadow_buffer.depth.width)
	su.rl.DrawTextureEx(shadow_buffer.texture, Vector2(WINDOW_w - display_size, 0.0), rotation, display_scale, su.rl.RAYWHITE)
	su.rl.DrawTextureEx(shadow_buffer2.texture, Vector2(WINDOW_w - display_size, display_size), rotation, display_scale, su.rl.RAYWHITE)
	su.rl.DrawTextureEx(shadow_buffer.depth, Vector2(WINDOW_w - display_size, 2 * display_size), rotation, display_scale, su.rl.RAYWHITE)
def draw_prepass():
	display_size = WINDOW_w / 5.0
	display_scale = display_size / float(prepass_buffer.texture.width)
	su.rl.DrawTextureEx(prepass_buffer.texture, Vector2(WINDOW_w - display_size, 0.0), rotation, display_scale, su.rl.RAYWHITE)
def draw_AO():
	# NOTE: this is after goin on the downsampling / upsampling roller-coaster
	# if you want the downsampling results you have to comment the upsampling
	display_size = WINDOW_w / 5.0
	display_scale = display_size / float(AO_buffer.texture.width)
	su.rl.DrawTextureEx(AO_buffer.texture, Vector2(WINDOW_w - display_size, 0), rotation, display_scale, su.rl.RAYWHITE)
	display_scale = display_size / float(AO_buffer2.texture.width)
	su.rl.DrawTextureEx(AO_buffer2.texture, Vector2(WINDOW_w - display_size, display_size), rotation, display_scale, su.rl.RAYWHITE)
	display_scale = display_size / float(AO_buffer3.texture.width)
	su.rl.DrawTextureEx(AO_buffer3.texture, Vector2(WINDOW_w - display_size, 2 * display_size), rotation, display_scale, su.rl.RAYWHITE)

orbit = True
applyAO = True
def inputs():
	global camera
	global camera_dist
	global unused_camera
	global orbit
	global applyAO

	if su.rl.IsKeyPressed(su.rl.KEY_R): load_shaders()
	if su.rl.IsKeyPressed(su.rl.KEY_O):
		orbit = not orbit
	if su.rl.IsKeyPressed(su.rl.KEY_I):
		if applyAO:
			with su.RenderContext(texture=AO_buffer) as render:
				su.ClearColorBuffer()
		applyAO = not applyAO

	if su.rl.IsKeyPressed(su.rl.KEY_P):
		light_camera.projection = (su.rl.CAMERA_ORTHOGRAPHIC
			if light_camera.projection == su.rl.CAMERA_PERSPECTIVE
			else su.rl.CAMERA_PERSPECTIVE
		)
	if su.rl.IsKeyPressed(su.rl.KEY_L):
		if unused_camera:
			camera = unused_camera
			unused_camera = None
		else:
			unused_camera = camera
			camera = light_camera

	scrollspeed = 3.0
	mw = scrollspeed * su.rl.GetMouseWheelMove()
	if mw != 0 and ( camera.position.y > scrollspeed + 0.5 or mw > 0.0):
		cam_pos = np.array([camera.position.x, camera.position.y, camera.position.z])
		tar = np.array([camera.target.x, camera.target.y, camera.target.z])
		camera_dist -= mw * 0.3;
		cam_pos[1] += (tar[1] - cam_pos[1]) / np.linalg.norm(cam_pos[1] + 0.1) * mw
		camera.position = Vector3(cam_pos[0], cam_pos[1], cam_pos[2])

def update(frameTime):
	global camera
	
	time = su.rl.GetTime()
	
	if orbit:
		cam_ang = time * 0.5
		camera.position = Vector3(
			np.cos(cam_ang) * camera_dist,
			camera.position.y,
			np.sin(cam_ang) * camera_dist)

	#global animFrameCounter
	#su.rl.UpdateModelAnimation(model, anims[0], animFrameCounter)
	#animFrameCounter += 1
	#if animFrameCounter >= anims[0].frameCount: animFrameCounter = 0

	pv = world.where(Position, Velocity)
	p_vec, v_vec = (positions.get_vector(pv), velocities.get_vector(pv))
	p_vec += v_vec * frameTime

	# bounce when out of bounds
	mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
	mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
	v_vec[mask_x, 0] *= -1
	v_vec[mask_z, 2] *= -1
	# if further than boundary, it would get stuck alternating direction each frame
	p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
	p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE

	positions.set_vector(pv, p_vec)
	velocities.set_vector(pv, v_vec)


def draw_scene(render:su.RenderContext, randomize_color=False):
	global model
	with render.shader:
		for i in range(model.materialCount):
			model.materials[i].shader = render.shader.shaderStruct
		for i in range(heightmap.materialCount):
			heightmap.materials[i].shader = render.shader.shaderStruct
		
		# model, position, rotation axis, rotation (deg), scale, tint
		#scale = 0.4
		#su.rl.DrawModelEx(model, Vector3(0,4,0), Vector3(1,0,0), 0.0, Vector3(scale,scale,scale), su.rl.BEIGE)
		scale = 1
		su.rl.DrawModelEx(model, Vector3(0,0,0), Vector3(1,0,0), 0.0, Vector3(scale,scale,scale), su.rl.BEIGE)
		su.rl.DrawModelEx(heightmap, Vector3(0,0,0), Vector3(1,0,0), 0.0, Vector3(scale,scale,scale), su.rl.BEIGE)		

		ents = world.where(Position, Mesh, BoundingBox)
		pos_vec, mesh_vec, bb_vec, = (positions.get_vector(ents), meshes.get_vector(ents), bboxes.get_vector(ents))
		bmins = bb_vec[:,:3] # entity (int), bounding box (6 floats)
		bmaxs = bb_vec[:,3:]
		sizes = bmaxs - bmins
		centers = pos_vec + (bmaxs + bmins) * 0.5
		meshIds = ents / np.max(ents)
		
		for meshId, center, size, mesh in zip(meshIds, centers, sizes, mesh_vec):
			su.rl.DrawCube(
				tuple(center),
				size[0], # x
				size[1], # y
				size[2], # z
				(int(255 * meshId),255,255,255) if randomize_color else mesh[meshes.color_id]
			)


run()