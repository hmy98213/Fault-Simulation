# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:52:03 2021

@author: hmy98
"""

import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit.algorithms.linear_solvers.matrices import TridiagonalToeplitz
from qiskit.extensions import UnitaryGate
from qiskit.tools.visualization import dag_drawer
from qiskit.algorithms.linear_solvers.hhl import *
from qiskit.circuit.library import *
from my import *
from cir_input.circuit_DG import QiskitCircuitToDG
from scipy import linalg
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
from qiskit.visualization import plot_histogram
import scipy
from numpy import linalg as LA
import myhhl


def paperhhl():

    t = 2  # This is not optimal; As an exercise, set this to the
           # value that will get the best results. See section 8 for solution.
    
    nqubits = 4  # Total number of qubits
    nb = 1  # Number of qubits representing the solution
    nl = 2  # Number of qubits representing the eigenvalues
    
    theta = 0  # Angle defining |b>
    
    a = 1  # Matrix diagonal
    b = -1/3  # Matrix off-diagonal
    
    # Initialize the quantum and classical registers
    qr = QuantumRegister(nqubits)
    
    # Create a Quantum Circuit
    qc = QuantumCircuit(qr)
    
    qrb = qr[0:nb]
    qrl = qr[nb:nb+nl]
    qra = qr[nb+nl:nb+nl+1]
    
    # State preparation. 
    qc.ry(2*theta, qrb[0])
        
    # QPE with e^{iAt}
    for qu in qrl:
        qc.h(qu)
    
    qc.p(a*t, qrl[0])
    qc.p(a*t*2, qrl[1])
    
    qc.u(b*t, -np.pi/2, np.pi/2, qrb[0])
    
    
    # Controlled e^{iAt} on \lambda_{1}:
    params=b*t
    
    qc.p(np.pi/2,qrb[0])
    qc.cx(qrl[0],qrb[0])
    qc.ry(params,qrb[0])
    qc.cx(qrl[0],qrb[0])
    qc.ry(-params,qrb[0])
    qc.p(3*np.pi/2,qrb[0])
    
    # Controlled e^{2iAt} on \lambda_{2}:
    params = b*t*2
    
    qc.p(np.pi/2,qrb[0])
    qc.cx(qrl[1],qrb[0])
    qc.ry(params,qrb[0])
    qc.cx(qrl[1],qrb[0])
    qc.ry(-params,qrb[0])
    qc.p(3*np.pi/2,qrb[0])
    
    # Inverse QFT
    qc.h(qrl[1])
    qc.rz(-np.pi/4,qrl[1])
    qc.cx(qrl[0],qrl[1])
    qc.rz(np.pi/4,qrl[1])
    qc.cx(qrl[0],qrl[1])
    qc.rz(-np.pi/4,qrl[0])
    qc.h(qrl[0])
    
    # Eigenvalue rotation
    t1=(-np.pi +np.pi/3 - 2*np.arcsin(1/3))/4
    t2=(-np.pi -np.pi/3 + 2*np.arcsin(1/3))/4
    t3=(np.pi -np.pi/3 - 2*np.arcsin(1/3))/4
    t4=(np.pi +np.pi/3 + 2*np.arcsin(1/3))/4
    
    qc.cx(qrl[1],qra[0])
    qc.ry(t1,qra[0])
    qc.cx(qrl[0],qra[0])
    qc.ry(t2,qra[0])
    qc.cx(qrl[1],qra[0])
    qc.ry(t3,qra[0])
    qc.cx(qrl[0],qra[0])
    qc.ry(t4,qra[0])
    #qc.measure_all()
    return qc

def myHHLpre():
    matrix = scipy.sparse.random(8, 8, 0.2).toarray()
    matrix = matrix + matrix.transpose()
    #print(matrix)
    print(LA.cond(matrix))
    vector = np.random.rand(8, 1)
    #naive_hhl_solution = HHL().solve(matrix, vector)
    #classical_solution = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector))
    qc = HHL().construct_circuit(matrix, vector)
    #print(qc)
    #print(qc.decompose())
    #print(qc.decompose().decompose())
    nqubits = qc.num_qubits
    qr = QuantumRegister(nqubits, 'q')
    circ = QuantumCircuit(qr)
    circ.append(qc, qr)
    for i in range(10):
        circ = circ.decompose()
    return circ

def my2HHL():
    #matrix = TridiagonalToeplitz(3, 1, 1 / 3, trotter_steps=2)
    matrix = scipy.sparse.random(8, 8, 0.2).toarray()
    matrix = (matrix + matrix.transpose())
    #matrix = np.array([[1, -1/3], [-1/3, 1]])
    unitary = linalg.expm(1.0j * matrix)
    vector = np.random.rand(8, 1)
    qc = myhhl.HHL().my_construct_circuit(unitary, vector)
    '''
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        print(node.name)
        qr = QuantumRegister(node.op.num_qubits)
        qc = QuantumCircuit(qr)
        qc.append(node.op, qr)
        this_qc = transpile(qc, basis_gates=['cx', 'u3'])
        if node.name == '1/x':
            print(this_qc)
        print(this_qc.size())
    #print(qc)
    '''
    nqubits = qc.num_qubits
    qr = QuantumRegister(nqubits, 'q')
    circ = QuantumCircuit(qr)
    circ.append(qc, qr)
    trans_qc = transpile(circ, basis_gates=['cx', 'u3'])
    #print(trans_qc)
    return trans_qc

if __name__ == '__main__':
    cir = my2HHL()
    #print(cir)
    res = QiskitCircuitToDG(cir, flag_single=True, flag_interaction=False)
    dag_cir = res[0]
    #cir.draw(output='mpl', filename='hhl.png')
    #print(cir)
    
    print(cir.depth())
    print(cir.size())
    dag = circuit_to_dag(cir)
    nqubits = cir.num_qubits
    error_num = 2
    error_pos = []
    error_pos = generate_error(cir.size(), error_num)
    
    #print('circuit:', file_name)
    
    
    num_qubit = get_real_qubit_num(dag_cir)
    print('qubits:', num_qubit)
    gate_num = get_gates_number(dag_cir)
    print('gates number:', gate_num)
    time_now = datetime.datetime.now()
    print(time_now.strftime('%m.%d-%H:%M:%S'))
    print('noisy_num:', error_num)
    t_start = time.time()
    #print(qc)
    #result = my_sim(dag_cir, error_pos)
    #print(result.data[-1][-1])
    run_time = time.time() - t_start
    print("simulate run time: ", run_time)
      
    #ps1=[0,0,0,0,0,0,0]
    #ps2=[0,1,0,0,0,0,0]
    ps1= [0 for i in range(nqubits)]
    ps2= [1 for i in range(nqubits)]
      
    t_start = time.time()
    result = reach1(cir, error_pos, ps1, ps2)
    run_time = time.time() - t_start
    print("alg1 run time: ", run_time)
    print(result)
      
    t_start = time.time()
    result = reach2(cir, error_pos, ps1, ps2)
    run_time = time.time() - t_start
    print("alg2 run time: ", run_time)
    print(result)
