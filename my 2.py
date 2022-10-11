import random
import time

import numpy as np
import tensornetwork as tn

from qiskit.converters import circuit_to_dag
from cir_input.qasm import CreateCircuitFromQASM

import warnings
warnings.filterwarnings("ignore")

CNOT = np.zeros((2, 2, 2, 2), dtype=complex)
CNOT[0][0][0][0] = 1
CNOT[0][1][0][1] = 1
CNOT[1][0][1][1] = 1
CNOT[1][1][1][0] = 1

def apply_gate(qubit_edges, gate, operating_qubits):
  op = tn.Node(gate)
  for i, bit in enumerate(operating_qubits):
    tn.connect(qubit_edges[bit], op[i])
    qubit_edges[bit] = op[i + len(operating_qubits)]


def example_in_paper():
  H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
  cS = np.zeros((2, 2, 2, 2), dtype=complex)
  cS[0][0][0][0] = 1
  cS[0][1][0][1] = 1
  cS[1][0][1][0] = 1
  cS[1][1][1][1] = 1.0j
  all_nodes=[]
  with tn.NodeCollection(all_nodes):
    start_gates = [
      tn.Node(np.eye(2, dtype=complex)) for _ in range(2)
    ]
    qubits = [node[1] for node in start_gates]
    start_wires = [node[0] for node in start_gates]
    apply_gate(qubits, cS, [0, 1])
    qubits[0]^start_wires[0]
    qubits[1]^start_wires[1]
  result = tn.contractors.optimal(
    all_nodes)
  print(result.tensor)  # array([0.707+0.j, 0.0+0.j], [0.0+0.j, 0.707+0.j])

def error_gate(k, p):
  if k==0: return np.eye(2, dtype=complex)*np.sqrt(1-p)
  elif k==1: return np.array([[0,1],[1,0]], dtype=complex)*np.sqrt(p/3)
  elif k==2: return np.array([[0,-1.0j],[1.0j,0]], dtype=complex)*np.sqrt(p/3)
  else: return np.array([[1,0],[0,-1]], dtype=complex)*np.sqrt(p/3)

def error_gate2(p):
  '''
  X = np.array([[0,1],[1,0]], dtype=complex)*np.sqrt(p/3)
  Y = np.array([[0,-1.0j],[1.0j,0]], dtype=complex)*np.sqrt(p/3)
  Z = np.array([[1,0],[0,-1]], dtype=complex)*np.sqrt(p/3)
  I = np.eye(2, dtype=complex)*np.sqrt(1-p)
  '''
  G=np.zeros((2,2,2,2), dtype= complex)
  G[0][0][0][0]=G[1][1][1][1]=1-2*p/3
  G[0][1][0][1]=G[1][0][1][0]=1-4*p/3
  G[0][0][1][1]=G[1][1][0][0]=2*p/3
  return G

def dag_to_error(dag, qubits, error_vec):
  pos=0
  for node in dag.topological_op_nodes():
    if node.name == 'cx':
      gate=CNOT
    else:
      gate=np.array(node.op.to_matrix(), dtype=complex)
    operating_qubits=[x.index for x in node.qargs]
    #print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
    apply_gate(qubits, gate, operating_qubits)
    if pos in error_vec:
      error_qubit = random.sample(operating_qubits, 1)
      apply_gate(qubits, error_gate(error_vec[pos], 0.001), error_qubit)
    pos = pos + 1

def dag_to_error2(dag, qubits0, qubits1, error_pos):
  pos=0
  for node in dag.topological_op_nodes():
    if node.name == 'cx':
      gate=CNOT
    else:
      gate=np.array(node.op.to_matrix(), dtype=complex)
    operating_qubits=[x.index for x in node.qargs]
    #print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
    apply_gate(qubits0, gate, operating_qubits)
    apply_gate(qubits1, gate.conjugate(), operating_qubits)
    if pos in error_pos:
      error_qubit = random.sample(operating_qubits, 1)[0]
      errors = [error_qubit, error_qubit + len(qubits0)]
      qubits = qubits0 + qubits1
      apply_gate(qubits, error_gate2(0.001), errors)
      qubits0[:]=qubits[:len(qubits0)]
      qubits1[:]=qubits[len(qubits0):]
    pos = pos + 1

def dag_to_tn(dag, qubits, flag=0):
  for node in dag.topological_op_nodes():
    if node.name == 'cx':
      gate=CNOT
    else:
      if flag == 0:
        gate=np.array(node.op.to_matrix(), dtype=complex)
      else:
        gate=np.array(node.op.to_matrix(), dtype=complex).conjugate()
    #print(gate)
    operating_qubits=[x.index for x in node.qargs]
    #print(node.name, node.qargs[:].index, operating_qubits, node.op.to_matrix())
    apply_gate(qubits, gate, operating_qubits)

def num_to_dic(n, error_pos):
  error_dic={}
  for i in range(len(error_pos)):
    q,r = divmod(n, 4)
    error_dic[error_pos[i]]=r
    n=q
  return error_dic

def generate_error(n, noisy_gate_num = 0):
  if noisy_gate_num == 0:
    noisy_gate_num = np.random.randint(1, min(15, n))
  list = [i for i in range(n)]
  return random.sample(list, noisy_gate_num)

def alg1(cir, error_pos):
  invcir = cir.inverse()
  error_num = len(error_pos)
  result = 0
  for n in range(4**error_num):
    all_nodes = []
    with tn.NodeCollection(all_nodes):
      start_gates = [
        tn.Node(np.eye(2, dtype=complex)) for _ in range(cir.num_qubits)
      ]
      qubits = [node[1] for node in start_gates]
      start_wires = [node[0] for node in start_gates]
      invdag = circuit_to_dag(invcir)
      dag_to_tn(invdag, qubits)
      dag = circuit_to_dag(cir)
      error_dic= num_to_dic(n, error_pos)
      dag_to_error(dag, qubits, error_dic)
      for i in range(cir.num_qubits):
        tn.connect(start_wires[i], qubits[i])
      tmp=tn.contractors.auto(all_nodes).tensor
      result = result + abs(tmp)**2
  result = result / 2**(cir.num_qubits * 2)
  #print(result)
  return result

def alg2(cir, error_pos):
  invcir = cir.inverse()
  all_nodes=[]
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
    invdag = circuit_to_dag(invcir)
    dag = circuit_to_dag(cir)
    dag_to_tn(invdag, qubits0)
    dag_to_tn(invdag, qubits1, 1)
    dag_to_error2(dag, qubits0, qubits1, error_pos)

    for i in range(cir.num_qubits):
      start_wires0[i] ^ qubits0[i]
      start_wires1[i] ^ qubits1[i]
  result = tn.contractors.auto(all_nodes).tensor
  result = result / 2**(cir.num_qubits * 2)
  result = np.real(result)
  #print(result)
  #print(len(error_pos))
  return result



if __name__ == '__main__':
  path = 'Benchmarks/'
  file_name = 'qft_n10.qasm'
  cir = CreateCircuitFromQASM(file_name, path)

  error_pos = generate_error(cir.size())
  alg1(cir, error_pos)
  t_start = time.time()

  alg2(cir, error_pos)
  run_time = time.time() - t_start
  print('run_time:', run_time)
  #example_in_paper()