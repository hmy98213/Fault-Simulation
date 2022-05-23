import numpy as np
import jax
import jax.numpy as jnp
import tensornetwork as tn
from jax import jit
import time

def calculate_abc_trace(a, b, c):
  an = tn.Node(a)
  bn = tn.Node(b)
  cn = tn.Node(c)
  an[1] ^ bn[0]
  bn[1] ^ cn[0]
  cn[1] ^ an[0]
  return tn.contractors.auto([an, bn, cn]).tensor

@jit
def multi_calculation(a, b, c):
  result = 0
  for i in range(10):
    if i<=5:
      result += calculate_abc_trace(a,b,c)
    else:
      result += calculate_abc_trace(c,b,a)
  return result

# a = np.ones((4096, 4096))
# b = np.ones((4096, 4096))
# c = np.ones((4096, 4096))
# tn.set_default_backend("numpy")
# print("Numpy Backend")
# start = time.time()
# np.array(multi_calculation(a, b, c))
# print("CPU: ", time.time()-start)

tn.set_default_backend("jax")
a = jnp.ones((4096, 4096))
b = jnp.ones((4096, 4096))
c =jnp.ones((4096, 4096))
print("JAX Backend")
start = time.time()
np.array(multi_calculation(a, b, c))
print("GPU: ", time.time()-start)
#
# fast_calculate = jit(multi_calculation)
# print("JIT Compile:")
# start = time.time()
# np.array(fast_calculate(a, b, c))
# print("GPU_JIT: ", time.time()-start)


