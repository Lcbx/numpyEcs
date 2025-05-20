from typing import Any, Sequence
from raylib import rlSetUniform, \
	RL_SHADER_UNIFORM_FLOAT, RL_SHADER_UNIFORM_INT, \
	RL_SHADER_UNIFORM_VEC2, RL_SHADER_UNIFORM_VEC3, RL_SHADER_UNIFORM_VEC4
from pyray import Vector2, Vector3, Vector4, ffi
#from weakref import WeakKeyDictionary

""" # in rlgl.h

# here value is of type void *
rlSetUniform(locIndex, value, uniformType, count);

# uniform types :
	case RL_SHADER_UNIFORM_FLOAT: glUniform1fv(locIndex, count, (float *)value); break;
    case RL_SHADER_UNIFORM_VEC2: glUniform2fv(locIndex, count, (float *)value); break;
    case RL_SHADER_UNIFORM_VEC3: glUniform3fv(locIndex, count, (float *)value); break;
    case RL_SHADER_UNIFORM_VEC4: glUniform4fv(locIndex, count, (float *)value); break;
    case RL_SHADER_UNIFORM_INT:  glUniform1iv(locIndex, count, (int *)value); break;
    case RL_SHADER_UNIFORM_IVEC2: glUniform2iv(locIndex, count, (int *)value); break;
    case RL_SHADER_UNIFORM_IVEC3: glUniform3iv(locIndex, count, (int *)value); break;
    case RL_SHADER_UNIFORM_IVEC4: glUniform4iv(locIndex, count, (int *)value); break;
#if !defined(GRAPHICS_API_OPENGL_ES2)
    case RL_SHADER_UNIFORM_UINT: glUniform1uiv(locIndex, count, (unsigned int *)value); break;
    case RL_SHADER_UNIFORM_UIVEC2: glUniform2uiv(locIndex, count, (unsigned int *)value); break;
    case RL_SHADER_UNIFORM_UIVEC3: glUniform3uiv(locIndex, count, (unsigned int *)value); break;
    case RL_SHADER_UNIFORM_UIVEC4: glUniform4uiv(locIndex, count, (unsigned int *)value); break;
#endif
    case RL_SHADER_UNIFORM_SAMPLER2D: glUniform1iv(locIndex, count, (int *)value); break;

"""

# helper to detect cdata struct type
def _struct_info(v):
    ty = ffi.typeof(v)
    if ty.kind == 'struct':
        name = ty.cname  # ex: 'struct Vector2'
        return name.replace('struct ', '')
    return None

def SetShaderValue(loc: int, value: Any) -> None:
	"""
	Upload `value` to the shader uniform at location `loc`.
	Supports:
	  - int or float scalars
	  - cffi structs for Vector2, Vector3, Vector4
	  - homogeneous sequences of ints, floats, or Vector2/3/4 structs
	"""

	if isinstance(value, int):
	    c_arr    = ffi.new("int *", value)
	    uni_type = RL_SHADER_UNIFORM_INT
	    count    = 1

	elif isinstance(value, float):
	    c_arr    = ffi.new("float *", value)
	    uni_type = RL_SHADER_UNIFORM_FLOAT
	    count    = 1

	else:
		struct_name = _struct_info(value)
		if struct_name in ('Vector2', 'Vector3', 'Vector4'):
		    # extract fields x,y,z,w as needed
		    dims = int(struct_name[-1])
		    coords = [getattr(value, axis) for axis in ('x','y','z','w')[:dims]]
		    c_arr    = ffi.new(f"float[{dims}]", coords)
		    uni_type = {
		        2: RL_SHADER_UNIFORM_VEC2,
		        3: RL_SHADER_UNIFORM_VEC3,
		        4: RL_SHADER_UNIFORM_VEC4,
		    }[dims]
		    count    = 1

		elif isinstance(value, Sequence):
			if not value: raise ValueError("Cannot upload an empty sequence as a uniform")
			first = value[0]
			seq_len = len(value)
			seq_struct_name = _struct_info(first)

			if isinstance(first, int):
			    c_arr    = ffi.new(f"int[{seq_len}]", value)
			    uni_type = RL_SHADER_UNIFORM_INT
			    count    = seq_len

			elif isinstance(first, float):
			    c_arr    = ffi.new(f"float[{seq_len}]", value)
			    uni_type = RL_SHADER_UNIFORM_FLOAT
			    count    = seq_len

			elif seq_struct_name in ('Vector2', 'Vector3', 'Vector4'):
			    dims = int(seq_struct_name[-1])
			    size = dims * seq_len
			    flat = []
			    flat.ensureCapacity(size)
			    for v in value:
			        flat.extend(getattr(v, axis) for axis in ('x','y','z','w')[:dims])
			    c_arr    = ffi.new(f"float[{size}]", flat)
			    uni_type = {
			        2: RL_SHADER_UNIFORM_VEC2,
			        3: RL_SHADER_UNIFORM_VEC3,
			        4: RL_SHADER_UNIFORM_VEC4,
			    }[dims]
			    count    = seq_len
			else:
				raise TypeError(f"Unsupported uniform inner array value type: {seq_struct_name}")
		else:
			raise TypeError(f"Unsupported uniform value type: {struct_name}")

	rlSetUniform(loc, c_arr, uni_type, count)
