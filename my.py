import random
import time

import numpy as np
from random import randrange
import tensornetwork as tn
import numba as nb
import jax.numpy as jnp
from qiskit import transpile, execute, QuantumCircuit, Aer

from qiskit.converters import circuit_to_dag
from qiskit.providers.aer import QasmSimulator

from cir_input.qasm import CreateCircuitFromQASM
from qiskit_fidelity import *
import jax
from jax import jit
from functools import partial
import warnings
from scipy.stats import unitary_group
import os

warnings.filterwarnings("ignore")


def generate_error(n, noisy_gate_num=0):
    l = [i for i in range(n)]
    return random.sample(l, noisy_gate_num)


def gen_all_basis(numq):
    result = [[]]
    for i in range(numq):
        result0 = copy.deepcopy(result)
        result1 = copy.deepcopy(result)
        for item in result0:
            item.append(0)
        for item in result1:
            item.append(1)
        result = result0 + result1
    return result


def arr_to_tnvec1(arr):
    vec = []
    for i in arr:
        vec.append(tn.Node(jnp.array([(1 + (-1) ** i) / 2 + 0.0j, (1 - (-1) ** i) / 2 + 0.0j])))
    return vec


def arr_to_tnvec(arr):
    vec = []
    for mat in arr:
        vec.append(tn.Node(jnp.array(mat)))
    return vec


CNOT = np.zeros((2, 2, 2, 2), dtype=complex)
CNOT[0][0][0][0] = 1
CNOT[0][1][0][1] = 1
CNOT[1][0][1][1] = 1
CNOT[1][1][1][0] = 1
CNOT = jnp.array(CNOT)

tn.set_default_backend("jax")  # tensorflow, pytorch, numpy, symmetric
path = 'Benchmarks/'
file_name = 'qft_n34.qasm'
cir, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)
dag_cir = res[0]
print('depth:', cir.depth())

dag = circuit_to_dag(cir)
nqubits = cir.num_qubits
error_num = randrange(min(cir.size(), 8))
# error_num = 2

# error_pos = [10, 20]
error_pos = generate_error(cir.size(), error_num)
error_pos.sort()

ps1 = [jnp.array([1.0, 0], dtype=complex) for i in range(nqubits)]
# set_ps2 = [[jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex) for i in range(nqubits)]]
ps2 = [jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex) for i in range(nqubits)]


# set_ps2 = [[jnp.array([0, 1.0], dtype = complex) for i in range(nqubits)]]
# set_ps2= gen_all_basis(nqubits)
# print(len(set_ps2))

def apply_gate(qubit_edges, gate, operating_qubits):
    op = tn.Node(gate)
    for i, bit in enumerate(operating_qubits):
        tn.connect(qubit_edges[bit], op[i])
        qubit_edges[bit] = op[i + len(operating_qubits)]


def error_gate(k, p):
    if k == 0:
        return jnp.eye(2, dtype=complex) * jnp.sqrt(1 - p)
    elif k == 1:
        return jnp.array([[0, 1], [1, 0]], dtype=complex) * jnp.sqrt(p / 3)
    elif k == 2:
        return jnp.array([[0, -1.0j], [1.0j, 0]], dtype=complex) * jnp.sqrt(p / 3)
    else:
        return jnp.array([[1, 0], [0, -1]], dtype=complex) * jnp.sqrt(p / 3)


def error_gate2(p):
    '''
  X = jnp.array([[0,1],[1,0]], dtype=complex)*jnp.sqrt(p/3)
  Y = jnp.array([[0,-1.0j],[1.0j,0]], dtype=complex)*jnp.sqrt(p/3)
  Z = jnp.array([[1,0],[0,-1]], dtype=complex)*jnp.sqrt(p/3)
  I = jnp.eye(2, dtype=complex)*jnp.sqrt(1-p)
  '''
    G = np.zeros((2, 2, 2, 2), dtype=complex)
    G[0][0][0][0] = G[1][1][1][1] = 1 - 2 * p / 3
    G[0][1][0][1] = G[1][0][1][0] = 1 - 4 * p / 3
    G[0][0][1][1] = G[1][1][0][0] = 2 * p / 3
    G = jnp.array(G)
    return G


def dag_to_error_unitary(dag, qubits):
    pos = 0
    cnt = 0
    for node in dag.topological_op_nodes():
        # print(node.op.qasm())
        if node.name == 'cx':
            gate = CNOT
        elif node.name.startswith('circuit'):
            continue
        else:
            # print(node.name)
            gate = np.array(node.op.to_matrix(), dtype=complex)
            if gate.size == 16:
                gate = gate.reshape(2, 2, 2, 2)
        operating_qubits = [x.index for x in node.qargs]
        # print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
        apply_gate(qubits, gate, operating_qubits)
        if cnt < error_num and pos == error_pos[cnt]:
            error_qubit = random.sample(operating_qubits, 1)
            error_g = np.array(unitary_group.rvs(2))
            apply_gate(qubits, error_g, error_qubit)
            cnt = cnt + 1
        pos = pos + 1


def dag_to_error(dag, qubits, error_vec):
    pos = 0
    for node in dag.topological_op_nodes():
        # print(node.op.qasm())
        if node.name == 'cx':
            gate = CNOT
        elif node.name.startswith('circuit'):
            continue
        else:
            # print(node.name)
            gate = jnp.array(node.op.to_matrix(), dtype=complex)
            if gate.size == 16:
                gate = gate.reshape(2, 2, 2, 2)
        operating_qubits = [x.index for x in node.qargs]
        # print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
        apply_gate(qubits, gate, operating_qubits)
        if pos in error_vec:
            error_qubit = random.sample(operating_qubits, 1)
            apply_gate(qubits, error_gate(error_vec[pos], 0.001), error_qubit)
        pos = pos + 1


def dag_to_error2(dag, qubits0, qubits1):
    pos = 0
    cnt = 0
    for node in dag.topological_op_nodes():
        if node.name == 'cx':
            gate = CNOT
        else:
            gate = jnp.array(node.op.to_matrix(), dtype=complex)
            if gate.size == 16:
                gate = gate.reshape(2, 2, 2, 2)
        operating_qubits = [x.index for x in node.qargs]
        # print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
        apply_gate(qubits0, gate, operating_qubits)
        apply_gate(qubits1, gate.conjugate(), operating_qubits)
        if cnt < error_num and pos == error_pos[cnt]:
            error_qubit = random.sample(operating_qubits, 1)[0]
            errors = [error_qubit, error_qubit + len(qubits0)]
            qubits = qubits0 + qubits1
            apply_gate(qubits, error_gate2(0.001), errors)
            qubits0[:] = qubits[:len(qubits0)]
            qubits1[:] = qubits[len(qubits0):]
            cnt = cnt + 1
        pos = pos + 1


def dag_to_tn(dag, qubits, flag=0):
    for node in dag.topological_op_nodes():
        if node.name == 'cx':
            gate = CNOT
        else:
            if flag == 0:
                gate = jnp.array(node.op.to_matrix(), dtype=complex)
            else:
                gate = jnp.array(node.op.to_matrix(), dtype=complex).conjugate()
        # print(gate)
        operating_qubits = [x.index for x in node.qargs]
        # print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
        apply_gate(qubits, gate, operating_qubits)


def num_to_dic(n):
    error_dic = {}
    for i in range(len(error_pos)):
        q, r = divmod(n, 4)
        error_dic[error_pos[i]] = r
        n = q
    return error_dic


@jit
def unitary_error(ps1, ps2):
    all_nodes = []
    with tn.NodeCollection(all_nodes):
        start_gates = [
            tn.Node(np.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
        ]
        qubits = [node[1] for node in start_gates]
        start_wires = [node[0] for node in start_gates]
        dag = circuit_to_dag(cir)
        dag_to_error_unitary(dag, qubits)

        left_vec = arr_to_tnvec(ps1)
        right_vec = arr_to_tnvec(ps2)

        for i in range(cir.num_qubits):
            tn.connect(start_wires[i], left_vec[i][0])
            tn.connect(qubits[i], right_vec[i][0])
    return tn.contractors.auto(all_nodes).tensor


def reach_unitary():
    result = unitary_error(ps1, ps2)
    return abs(result) ** 2

# @jit
# def one_shot1(ps1, ps2, n):
#     all_nodes = []
#     with tn.NodeCollection(all_nodes):
#         start_gates = [
#             tn.Node(np.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
#         ]
#         qubits = [node[1] for node in start_gates]
#         start_wires = [node[0] for node in start_gates]
#         dag = circuit_to_dag(cir)
#         error_dic = num_to_dic(n)
#         dag_to_error(dag, qubits, error_dic)
#
#         left_vec = arr_to_tnvec(ps1)
#         right_vec = arr_to_tnvec(ps2)
#
#         for i in range(cir.num_qubits):
#             tn.connect(start_wires[i], left_vec[i][0])
#             tn.connect(qubits[i], right_vec[i][0])
#     return tn.contractors.auto(all_nodes).tensor


# @jit
# def reach1():
#     error_num = len(error_pos)
#     result = 0
#     for n in range(4 ** error_num):
#         tmp = 0
#         for ps2 in set_ps2:
#             tmp += one_shot1(ps1, ps2, n)
#         result += abs(tmp) ** 2
#     return result


@jit
def one_shot2(ps1, ps20, ps21):
    left_vec = arr_to_tnvec(ps1)
    right_vec0 = arr_to_tnvec(ps20)
    right_vec1 = arr_to_tnvec(ps21)
    left_vec0 = copy.deepcopy(left_vec)
    left_vec1 = copy.deepcopy(left_vec)

    all_nodes = []

    with tn.NodeCollection(all_nodes):
        start_gates0 = [
            tn.Node(np.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
        ]
        start_gates1 = [
            tn.Node(np.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
        ]
        qubits0 = [node[1] for node in start_gates0]
        qubits1 = [node[1] for node in start_gates1]
        start_wires0 = [node[0] for node in start_gates0]
        start_wires1 = [node[0] for node in start_gates1]
        dag = circuit_to_dag(cir)
        dag_to_error2(dag, qubits0, qubits1)

        for i in range(cir.num_qubits):
            tn.connect(start_wires0[i], left_vec0[i][0])
            tn.connect(qubits0[i], right_vec0[i][0])
            tn.connect(start_wires1[i], left_vec1[i][0])
            tn.connect(qubits1[i], right_vec1[i][0])
    return tn.contractors.auto(all_nodes + left_vec0 + left_vec1 + right_vec0 + right_vec1).tensor


# def reach2():
#     result = 0
#     for ps20 in set_ps2:
#         for ps21 in set_ps2:
#             result += one_shot2(ps1, ps20, ps21)
#     return np.real(result)


# def reach22(cir):
#   total = 0
#   for ps2 in set_ps2:
#     all_nodes = []
#     with tn.NodeCollection(all_nodes):
#       start_gates0 = [
#         tn.Node(jnp.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
#       ]
#       start_gates1 = [
#         tn.Node(jnp.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
#       ]
#       qubits0 = [node[1] for node in start_gates0]
#       qubits1 = [node[1] for node in start_gates1]
#       start_wires0 = [node[0] for node in start_gates0]
#       start_wires1 = [node[0] for node in start_gates1]
#       dag = circuit_to_dag(cir)
#       dag_to_error2(dag, qubits0, qubits1)
#
#       left_vec0 = []
#       left_vec1 = []
#       right_vec0 = []
#       right_vec1 = []
#
#       for i in ps1:
#         if i == 0:
#           left_vec0.append(tn.Node(jnp.array([1.0 + 0.0j, 0.0 + 0.0j])))
#         else:
#           left_vec0.append(tn.Node(jnp.array([0.0 + 0.0j, 1.0 + 0.0j])))
#       for i in ps1:
#         if i == 0:
#           left_vec1.append(tn.Node(jnp.array([1.0 + 0.0j, 0.0 + 0.0j])))
#         else:
#           left_vec1.append(tn.Node(jnp.array([0.0 + 0.0j, 1.0 + 0.0j])))
#       for i in ps2:
#         if i == 0:
#           right_vec0.append(tn.Node(jnp.array([1.0 + 0.0j, 0.0 + 0.0j])))
#         else:
#           right_vec0.append(tn.Node(jnp.array([0.0 + 0.0j, 1.0 + 0.0j])))
#       for i in ps2:
#         if i == 0:
#           right_vec1.append(tn.Node(jnp.array([1.0 + 0.0j, 0.0 + 0.0j])))
#         else:
#           right_vec1.append(tn.Node(jnp.array([0.0 + 0.0j, 1.0 + 0.0j])))
#
#       for i in range(cir.num_qubits):
#         tn.connect(start_wires0[i], left_vec0[i][0])
#         tn.connect(qubits0[i], right_vec0[i][0])
#         tn.connect(start_wires1[i], left_vec1[i][0])
#         tn.connect(qubits1[i], right_vec1[i][0])
#     result = tn.contractors.auto(all_nodes).tensor
#     total += result
#     # print(result)
#     # print(len(error_pos))
#   return jnp.real(result)


if __name__ == '__main__':
    print('circuit:', file_name)
    num_qubit = get_real_qubit_num(dag_cir)
    print('qubits:', num_qubit)
    gate_num = get_gates_number(dag_cir)
    print('gates number:', gate_num)
    time_now = datetime.datetime.now()
    print(time_now.strftime('%m.%d-%H:%M:%S'))

    print('noisy_num:', len(error_pos))

    # t_start = time.time()
    # print(qc)
    # result = my_sim(dag_cir, error_pos)
    # print(result.data[-1][-1])
    # run_time = time.time() - t_start
    # print("simulate run time: ", run_time)

    # ps1=[0,0,0,0,0,0,0]
    # ps2=[0,1,0,0,0,0,0]

    # t_start = time.time()
    # result = reach1()
    # run_time = time.time() - t_start
    # print("alg1 run time: ", run_time)
    # print(np.sqrt(result))

    # t_start = time.time()
    # result = one_shot2(ps1, ps2, ps2)
    # run_time = time.time() - t_start
    # print("alg2 run time: ", run_time)
    # print(np.sqrt(result))

    t_start = time.time()
    result = reach_unitary()
    run_time = time.time() - t_start
    print("unitary error run time: ", run_time)
    print(np.sqrt(result))
