import tensornetwork as tn
import numpy as np

def matrix_swap(M):
  return np.transpose(M, (1, 0, 3, 2))

CNOT = np.zeros((4, 4), dtype=complex)
CNOT[0][0] = 1.1j
CNOT[1][1] = 1
CNOT[2][3] = 1
CNOT[3][2] = 1
print(CNOT)
CNOT = CNOT.reshape(2,2,2,2)
print(CNOT)
CNOT = matrix_swap(CNOT)
CNOT = CNOT.reshape(4, 4)
print(CNOT)

