import gc
import random
import time

import numpy as np
from random import randrange
import tensornetwork as tn
from qiskit import transpile, execute, QuantumCircuit, Aer

from qiskit.converters import circuit_to_dag
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import Statevector, state_fidelity

from cir_input.qasm import CreateCircuitFromQASM
from qiskit_fidelity import *
import warnings
from scipy.stats import unitary_group

warnings.filterwarnings("ignore")


def generate_error(n, noisy_gate_num=0):
  l = [i for i in range(n)]
  return random.sample(l, noisy_gate_num)

def arr_to_tnvec(arr):
  vec = []
  for mat in arr:
    vec.append(tn.Node(np.array(mat)))
  return vec

def matrix_swap(M):
  return np.transpose(M, (1, 0, 3, 2))

CNOT = np.zeros((2, 2, 2, 2), dtype=complex)
CNOT[0][0][0][0] = 1
CNOT[0][1][0][1] = 1
CNOT[1][0][1][1] = 1
CNOT[1][1][1][0] = 1
CNOT = np.array(CNOT)

real_CNOT = np.array([[0.99998*np.exp(0.67*1.0j), 0.00271*np.exp(0.79*1.0j),0.00443*np.exp(-0.90*1.0j),0.00273*np.exp(-2.59*1.0j)],
                      [0.00271*np.exp(0.79*1.0j), 0.00357*np.exp(2.32*1.0j),0.00531*np.exp(-1.44*1.0j),0.99997*np.exp(0.67*1.0j)],
                      [0.00443*np.exp(-0.90*1.0j), 0.00531*np.exp(-1.44*1.0j),0.99996*np.exp(0.67*1.0j),0.00532*np.exp(-0.35*1.0j)],
                      [0.00273*np.exp(-2.59*1.0j), 0.99997*np.exp(-0.67*1.0j),0.00533*np.exp(-0.35*1.0j),0.00373*np.exp(2.16*1.0j)]
                      ], dtype=complex)
real_CNOT = real_CNOT.reshape(2, 2, 2, 2)
real_CNOT = matrix_swap(real_CNOT)

error_CNOT = np.zeros((6, 2, 2, 2, 2), dtype=complex)
TCF = np.array([[0.998*np.exp(1.56*1.0j), 0.048*np.exp(2.30*1.0j),0.014*np.exp(0.06*1.0j),0.049*np.exp(-2.19*1.0j)],
                [0.048*np.exp(2.30*1.0j), 0.042*np.exp(1.86*1.0j),0.012*np.exp(0.56*1.0j),0.998*np.exp(1.65*1.0j)],
                [0.014*np.exp(0.06*1.0j), 0.012*np.exp(0.56*1.0j),1.0*np.exp(1.68*1.0j),0.011*np.exp(-0.39*1.0j)],
                [0.049*np.exp(-2.19*1.0j), 0.998*np.exp(1.65*1.0j),0.011*np.exp(-0.39*1.0j),0.041*np.exp(-1.64*1.0j)]
                ], dtype=complex)
TCF = TCF.reshape(2, 2, 2, 2)
TCF = matrix_swap(TCF)

GSF = np.array([[0.98*np.exp(0.75j),0.11*np.exp(0.53j),0.11*np.exp(-0.87j),0.13*np.exp(-2.24j)],     [0.11*np.exp(0.53j),0.04*np.exp(-2.59j),0.25*np.exp(-1.18j),0.96*np.exp(0.66j)],[0.11*np.exp(-0.87j),0.25*np.exp(-1.18j),0.93*np.exp(0.62j),0.24*np.exp(-0.70j)],[0.13*np.exp(-2.24j),0.96*np.exp(0.66j),0.24*np.exp(-0.70j),0.02*np.exp(-0.84j)]],dtype=complex)
GSF = GSF.reshape(2, 2, 2, 2)
GSF = matrix_swap(GSF)

FDF = np.array([[0.97*np.exp(0.25j),0.13*np.exp(-2.25j),0.16*np.exp(1.88j),0.12*np.exp(-0.27j)],     [0.13*np.exp(-2.25j),0.18*np.exp(-3.09j),0.25*np.exp(1.86j),0.94*np.exp(0.83j)],[0.16*np.exp(1.88j),0.25*np.exp(1.86j),0.91*np.exp(0.78j),0.30*np.exp(2.79j)],[0.12*np.exp(-0.27j),0.94*np.exp(0.83j),0.30*np.exp(2.79j),0.10*np.exp(1.33j)]],dtype=complex)
FDF = FDF.reshape(2, 2, 2, 2)
FDF = matrix_swap(FDF)

RFF = np.array([[1.00*np.exp(-0.56j),0.00*np.exp(-0.43j),0.00*np.exp(-2.13j),0.00*np.exp(2.45j)],     [0.00*np.exp(0.80j),0.00*np.exp(2.31j),0.01*np.exp(-1.44j),1.00*np.exp(0.67j)],[0.00*np.exp(-0.90j),0.01*np.exp(-1.44j),1.00*np.exp(0.67j),0.01*np.exp(-0.35j)],[0.00*np.exp(-1.37j),1.00*np.exp(1.90j),0.01*np.exp(0.87j),0.00*np.exp(-2.88j)]],dtype=complex)
RFF = RFF.reshape(2, 2, 2, 2)
RFF = matrix_swap(RFF)

LFF1 = np.array([[0.99*np.exp(-0.11j),0.16*np.exp(-2.60j),0.06*np.exp(1.46j),0.06*np.exp(-0.73j)],     [0.06*np.exp(-1.97j),0.07*np.exp(-2.94j),0.12*np.exp(1.87j),0.99*np.exp(0.72j)],[0.06*np.exp(2.08j),0.12*np.exp(1.87j),0.98*np.exp(0.72j),0.13*np.exp(2.68j)],[0.06*np.exp(0.53j),0.99*np.exp(1.35j),0.13*np.exp(-2.98j),0.05*np.exp(1.72j)]],dtype=complex)
LFF1 = LFF1.reshape(2, 2, 2, 2)
LFF1 = matrix_swap(LFF1)

LFF2 = np.array([[0.99*np.exp(0.17j),0.07*np.exp(0.87j),0.06*np.exp(-1.35j),0.07*np.exp(2.61j)],     [0.07*np.exp(1.50j),0.05*np.exp(0.80j),0.15*np.exp(-1.25j),0.99*np.exp(0.63j)],[0.06*np.exp(-0.72j),0.15*np.exp(-1.25j),0.98*np.exp(0.62j),0.14*np.exp(-0.58j)],[0.07*np.exp(-2.42j),0.99*np.exp(1.26j),0.14*np.exp(0.05j),0.07*np.exp(-1.74j)]],dtype=complex)
LFF2 = LFF2.reshape(2, 2, 2, 2)
LFF2 = matrix_swap(LFF2)

Realistic_SMGF = np.array([[1.000*np.exp(0.00j),0.000*np.exp(0.00j),0.000*np.exp(0.00j),0.000*np.exp(0.00j)],     [0.000*np.exp(0.00j),0.996*np.exp(-2.71j),0.090*np.exp(1.34j),0.000*np.exp(0.00j)],[0.000*np.exp(0.00j),0.090*np.exp(1.34j),0.996*np.exp(2.25j),0.000*np.exp(0.00j)],[0.000*np.exp(0.00j),0.000*np.exp(0.00j),0.000*np.exp(0.00j),1.000*np.exp(0.00j)]],dtype=complex)
Realistic_SMGF = Realistic_SMGF.reshape(2, 2, 2, 2)
Realistic_SMGF = matrix_swap(Realistic_SMGF)

error_CNOT[0][:][:][:][:] = TCF
error_CNOT[1][:][:][:][:] = GSF
error_CNOT[2][:][:][:][:] = FDF
error_CNOT[3][:][:][:][:] = RFF
error_CNOT[4][:][:][:][:] = LFF1
error_CNOT[5][:][:][:][:] = LFF2

# ps2 = [np.array([0, 1.0], dtype = complex) for i in range(nqubits)]
# set_ps2= gen_all_basis(nqubits)
# print(len(set_ps2))

def apply_gate(qubit_edges, gate, operating_qubits):
  op = tn.Node(gate)
  for i, bit in enumerate(operating_qubits):
    tn.connect(qubit_edges[bit], op[i])
    qubit_edges[bit] = op[i + len(operating_qubits)]


def error_gate(k, p):
  if k == 0:
    return np.eye(2, dtype=complex) * np.sqrt(1 - p)
  elif k == 1:
    return np.array([[0, 1], [1, 0]], dtype=complex) * np.sqrt(p / 3)
  elif k == 2:
    return np.array([[0, -1.0j], [1.0j, 0]], dtype=complex) * np.sqrt(p / 3)
  else:
    return np.array([[1, 0], [0, -1]], dtype=complex) * np.sqrt(p / 3)


def error_gate2(p):
  choice = randrange(4)
  # depolarizing
  if choice == 0:
    G = np.zeros((2, 2, 2, 2), dtype=complex)
    G[0][0][0][0] = G[1][1][1][1] = 1 - 2 * p / 3
    G[0][1][0][1] = G[1][0][1][0] = 1 - 4 * p / 3
    G[0][0][1][1] = G[1][1][0][0] = 2 * p / 3
    G = np.array(G)
  # bit flip
  elif choice == 1:
    G = np.zeros((2, 2, 2, 2), dtype=complex)
    G[0][0][0][0] = G[0][1][0][1] = G[1][0][1][0] = G[1][1][1][1] = 1 - p
    G[0][0][1][1] = G[0][1][1][0] = G[1][0][0][1] = G[1][1][0][0] = p
    G = np.array(G)
  # phase flip
  elif choice == 2:
    G = np.zeros((2, 2, 2, 2), dtype=complex)
    G[0][0][0][0] =  G[1][1][1][1] = 1
    G[0][1][0][1] = G[1][0][1][0] = 1 - 2 * p
    G = np.array(G)
  # bit-phase flip
  elif choice == 3:
    G = np.zeros((2, 2, 2, 2), dtype=complex)
    G[0][0][0][0] = G[0][1][0][1] = G[1][0][1][0] = G[1][1][1][1] = 1 - p
    G[0][0][1][1] = G[1][1][0][0] = -p
    G[0][1][1][0] = G[1][0][0][1] = p
    G = np.array(G)
  return G


def dag_to_error_unitary(dag, qubits, error_num, case):
  pos = 0
  cnt = 0
  for node in dag.topological_op_nodes():
    # print(node.op.qasm())
    if node.name == 'cx':
      if cnt < error_num and case == 1:
        fault_id = randrange(2)
        gate = error_CNOT[fault_id]
        cnt += 1
      elif cnt < error_num and case == 0:
        gate = Realistic_SMGF
        cnt += 1
      else:
        gate = real_CNOT
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
      gate = np.array(node.op.to_matrix(), dtype=complex)
      if gate.size == 16:
        gate = gate.reshape(2, 2, 2, 2)
    operating_qubits = [x.index for x in node.qargs]
    # print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
    apply_gate(qubits, gate, operating_qubits)
    if pos in error_vec:
      error_qubit = random.sample(operating_qubits, 1)
      apply_gate(qubits, error_gate(error_vec[pos], 0.001), error_qubit)
    pos = pos + 1


def dag_to_error2(dag, qubits0, qubits1, error_num, error_pos):
  pos = 0
  cnt = 0
  for node in dag.topological_op_nodes():
    if node.name == 'cx':
      gate = real_CNOT
    else:
      gate = np.array(node.op.to_matrix(), dtype=complex)
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
        gate = np.array(node.op.to_matrix(), dtype=complex)
      else:
        gate = np.array(node.op.to_matrix(), dtype=complex).conjugate()
    # print(gate)
    operating_qubits = [x.index for x in node.qargs]
    # print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
    apply_gate(qubits, gate, operating_qubits)


def num_to_dic(n, error_pos):
  error_dic = {}
  for i in range(len(error_pos)):
    q, r = divmod(n, 4)
    error_dic[error_pos[i]] = r
    n = q
  return error_dic


def unitary_error(cir, ps1, ps2, error_num, case):
  all_nodes = []
  with tn.NodeCollection(all_nodes):
    start_gates = [
      tn.Node(np.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
    ]
    qubits = [node[1] for node in start_gates]
    start_wires = [node[0] for node in start_gates]
    dag = circuit_to_dag(cir)
    dag_to_error_unitary(dag, qubits, error_num, case)

    left_vec = arr_to_tnvec(ps1)
    right_vec = arr_to_tnvec(ps2)

    for i in range(cir.num_qubits):
      tn.connect(start_wires[i], left_vec[i][0])
      tn.connect(qubits[i], right_vec[i][0])
  return tn.contractors.auto(all_nodes).tensor


def reach_unitary(cir, ps1, ps2, error_num, case):
  result = unitary_error(cir, ps1, ps2, error_num, case)
  return abs(result) ** 2


def one_shot1(cir, ps1, ps2, n):
  all_nodes = []
  with tn.NodeCollection(all_nodes):
    start_gates = [
      tn.Node(np.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
    ]
    qubits = [node[1] for node in start_gates]
    start_wires = [node[0] for node in start_gates]
    dag = circuit_to_dag(cir)
    error_dic = num_to_dic(n)
    dag_to_error(dag, qubits, error_dic)

    left_vec = arr_to_tnvec(ps1)
    right_vec = arr_to_tnvec(ps2)

    for i in range(cir.num_qubits):
      tn.connect(start_wires[i], left_vec[i][0])
      tn.connect(qubits[i], right_vec[i][0])
  return tn.contractors.auto(all_nodes).tensor


def reach1():
  error_num = len(error_pos)
  result = 0
  for n in range(4 ** error_num):
    tmp = 0
    for ps2 in set_ps2:
      tmp += one_shot1(ps1, ps2, n)
    result += abs(tmp) ** 2
  return result


def one_shot2(cir, ps1, ps20, ps21, error_num, error_pos):
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
    dag_to_error2(dag, qubits0, qubits1, error_num, error_pos)

    for i in range(cir.num_qubits):
      tn.connect(start_wires0[i], left_vec0[i][0])
      tn.connect(qubits0[i], right_vec0[i][0])
      tn.connect(start_wires1[i], left_vec1[i][0])
      tn.connect(qubits1[i], right_vec1[i][0])
  return tn.contractors.auto(all_nodes + left_vec0 + left_vec1 + right_vec0 + right_vec1).tensor

def auto_run(path, file_name, output):
  f = open(output, 'a')
  f.write("\n")
  time_now = datetime.datetime.now()
  print(time_now.strftime('%m.%d-%H:%M:%S'))

  cir, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)
  dag_cir = res[0]
  nqubits = cir.num_qubits
  error_num = randrange(1, min(cir.size(), 8))
  # error_num = 0

  # error_pos = [10, 20]
  error_pos = generate_error(cir.size(), error_num)
  error_pos.sort()

  ps1q = [np.array([1.0, 0], dtype=complex) for i in range(nqubits)]
  ps1b = [np.array([1.0, 0], dtype=complex) for i in range(nqubits)]
  #ps1b[-1] = np.array([1/ np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)
  if file_name[0] == 'q':
    ps2q = [np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex) for i in range(nqubits)]
  elif file_name[0] == 'b':
    ps2b = [np.array([0, 1.0], dtype = complex) for i in range(nqubits)]
    ps2b[-1] = np.array([1/ np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)
  file_name = file_name.replace('.qasm', '')
  print('circuit:', file_name)
  f.write(file_name+"\t")

  num_qubit = get_real_qubit_num(dag_cir)
  print('qubits:', num_qubit)
  f.write(str(num_qubit)+"\t")

  gate_num = get_gates_number(dag_cir)
  print('gates number:', gate_num)
  f.write(str(gate_num)+"\t")

  print('depth:', cir.depth())
  f.write(str(cir.depth())+"\t")

  print('noisy_num:', len(error_pos))
  f.write(str(len(error_pos))+"\t")

  # t_start = time.time()
  # rho = my_sim(dag_cir, error_pos)
  # if file_name[0] == 'q':
  #     label_str = '+'* num_qubit
  # elif file_name[0] == 'b':
  #     label_str = '-' + '1'* (num_qubit - 1)
  # ps = Statevector.from_label(label_str)
  # result = np.sqrt(state_fidelity(ps, rho))
  # print(result)
  # run_time = time.time() - t_start
  # print("simulate run time: ", run_time)
  # f.write(str(run_time))

  t_start = time.time()
  rho = my_sim_unitary(cir)
  if file_name[0] == 'q':
    label_str = '+'* num_qubit
  elif file_name[0] == 'b':
    label_str = '-' + '1'* (num_qubit - 1)
  ps = Statevector.from_label(label_str)
  result = np.sqrt(state_fidelity(ps, rho))
  print(result)
  run_time = time.time() - t_start
  print("simulate run time: ", run_time)


# t_start = time.time()
  # result = reach1()
  # run_time = time.time() - t_start
  # print("alg1 run time: ", run_time)
  # print(np.sqrt(result))

  # t_start = time.time()
  # if file_name[0] == 'b':
  #   result = np.real(one_shot2(cir, ps1b, ps2b, ps2b, error_num, error_pos))
  # else:
  #   result = np.real(one_shot2(cir, ps1q, ps2q, ps2q, error_num, error_pos))
  # result = np.sqrt(result)
  # run_time = time.time() - t_start
  # print("alg2 run time: ", run_time)
  # f.write(str(run_time)+"\t")
  # print(np.sqrt(result))
  # f.write(str(result)+"\t")
  #
  #
  # t_start = time.time()
  # if file_name[0] == 'b':
  #   result = reach_unitary(cir, ps1b, ps2b, error_num, 0)
  # else:
  #   result = reach_unitary(cir, ps1q, ps2q, error_num, 0)
  # result = np.sqrt(result)
  # run_time = time.time() - t_start
  # print("Design Errors run time: ", run_time)
  # f.write(str(run_time)+"\t")
  # print(np.sqrt(result))
  # f.write(str(result)+"\t")
  #
  # t_start = time.time()
  # if file_name[0] == 'b':
  #   result = reach_unitary(cir, ps1b, ps2b, error_num, 1)
  # else:
  #   result = reach_unitary(cir, ps1q, ps2q, error_num, 1)
  # result = np.sqrt(result)
  # run_time = time.time() - t_start
  # print("Manufacturing Defects run time: ", run_time)
  # f.write(str(run_time)+"\t")
  # print(np.sqrt(result))
  # f.write(str(result)+"\t")


if __name__ == '__main__':
  path = 'Benchmarks/'
  files = os.listdir(path)
  file_name = 'qft_n3.qasm'
  output = 'final_result_sim.txt'
  auto_run(path, file_name, output)
  f = open(output, 'a')
  time_now = datetime.datetime.now()
  f.write(str(time_now))
  # for f in files:
  #   try:
  #     output = 'final_result_sim.txt'
  #     auto_run(path, f, output)
  #   except:
  #     pass
  #   gc.collect()
