import multiprocessing as mp
import numpy as np

from utils.conventions import *
import jax

class SharedArray(object):
    def __init__(self, shape, dtype):
        cdtype = np.ctypeslib.as_ctypes_type(dtype)
        self.share_mem = mp.RawArray(cdtype, int(np.prod(shape)))
        self.array = np.frombuffer(self.share_mem, dtype=dtype).reshape(shape)

    def re_init(self): # for spawn start method
        array = np.copy(self.array)
        self.array = np.frombuffer(self.share_mem, dtype=array.dtype).reshape(array.shape)
        np.copyto(self.array, array)


    def copyto(self, array):
        np.copyto(array, self.array)

    def get(self):
        return np.copy(self.array)

    def set(self, array):
        np.copyto(self.array, array)

class SharedJaxParams(object):
    def __init__(self, params):
        params = to_np(params)
        self.params = jax.tree_map(lambda p : SharedArray(p.shape, p.dtype), params)
        jax.tree_map(lambda s, a : s.set(a), self.params, params)

    def re_init(self):
        jax.tree_map(lambda p : p.re_init(), self.params)

    def get(self):
        return jax.tree_map(lambda p : p.get(), self.params)

    def set(self, params):
        jax.tree_map(lambda s, a : s.set(a), self.params, to_np(params))

    def copyto(self, params):
        jax.tree_map(lambda s, a : s.copyto(a), self.params, params)

class FakeSharedJaxParams:
    def __init__(self, params):
        self.params = params

    def set(self, params):
        self.params = params

    def get(self):
        return self.params

    def re_init(self):
        pass