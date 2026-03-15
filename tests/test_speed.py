import numpy as np
import time

# Simulate mesh
n_nodes = 10000
n_time_steps = 300

coords2 = np.random.rand(n_nodes, 2)
coords1 = np.random.rand(n_nodes, 1)
t_steps = np.linspace(0, 1, n_time_steps)

print(coords1.ndim)

def f(x, y, t):
    return np.sin(x) + y*t

def gscalar(x, y):
    return np.sin(x) + y

def hscalar(x):
    return np.sin(x)

startsc2 = time.time()
vec2d = gscalar(coords2[0, :], coords2[1, :])
endsc2 = time.time()

print(vec2d)

startsc1 = time.time()
vec1d = hscalar(coords1[0, :])
endsc1 = time.time()

print(vec1d)

startsct = time.time()
vect = f(*coords2.T, t_steps[:, None]).T
endsct = time.time()

print('here')
print(vect.shape)

start = time.time()
vec = f(coords2[0, :], coords2[1, :], t_steps[:, None])
end = time.time()

print('there')
print(vec.shape)

startr = time.time()
coord_args = tuple(coords2[:, i][:, None] for i in range(coords2.shape[1]))
vecr = f(*coord_args, t_steps[None, :])
endr = time.time()


print("Scalar 2D:", endsc2-startsc2, "s")
print("Scalar 1d:", endsc1-startsc1, "s")
print("Time:", endsct-startsct, "s")
print("Time (ch):", end-start, "s")
print("Time (r):", endr-startr, "s")