# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:03:50 2021

@author: hmy98
"""

import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit.algorithms.linear_solvers.hhl import HHL
matrix = np.array([[1, -1/3], [-1/3, 1]])
vector = np.array([1, 0])
naive_hhl_solution = HHL().solve(matrix, vector)
classical_solution = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector))
qc = HHL().construct_circuit(matrix, vector)
print(qc)



