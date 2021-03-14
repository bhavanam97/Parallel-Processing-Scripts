import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc
linalg.init()

# dot-product of 1D vector
a = np.asarray(np.random.rand(4, 2), np.float32)
b = np.asarray(np.random.rand(2, 2), np.float32)
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = linalg.dot(a_gpu, b_gpu)
print(np.allclose(np.dot(a, b), c_gpu.get()))

# dot-product of 2D vector
d = np.asarray(np.random.rand(5), np.float32)
e = np.asarray(np.random.rand(5), np.float32)
d_gpu = gpuarray.to_gpu(d)
e_gpu = gpuarray.to_gpu(e)
f = linalg.dot(d_gpu, e_gpu)
print(np.allclose(np.dot(d, e), f))
