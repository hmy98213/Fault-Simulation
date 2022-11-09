import numpy as np
import math

def cRz(theta):
    mat = np.zeros((2, 2, 2, 2), dtype=complex)
    mat[0][0][0][0] = 1
    mat[0][1][0][1] = 1
    mat[1][0][1][0] = np.exp(-1.0j * theta /2)
    mat[1][1][1][1] = np.exp(1.0j * theta /2)
    return np.array(mat)

def gen_noise_gate(T1, Tphi, t):
    a = math.exp(-t/T1)
    b = math.exp(-t/Tphi)
    G1 = np.matrix([[1,0],[0,math.sqrt(a)]])
    G2 = np.matrix([[0,math.sqrt(1-a)],[0,0]])

    F1 = np.matrix([[math.sqrt(b),0],[0,math.sqrt(b)]])
    F2 = np.matrix([[math.sqrt(1-b),0],[0,0]])
    F3 = np.matrix([[0,0],[0,math.sqrt(1 - b)]])

    E1 = F1 * G1
    E2 = F2 * G1
    E3 = F3 * G1
    E4 = F1 * G2
    E5 = F2 * G2
    E6 = F3 * G2

    E = [E1, E2, E3, E4, E5, E6]

    res = np.zeros((4, 4), dtype=complex)
    for e in E:
        res += np.kron(e, e.H)
    return np.reshape(res, (2, 2, 2, 2))