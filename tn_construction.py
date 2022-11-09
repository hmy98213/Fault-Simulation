from qiskit import QuantumCircuit
import math
import random
import numpy as np
import tensornetwork as tn
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.transpiler.passes import RemoveBarriers
class TimeoutException(Exception): pass
from error_gen import *

def file_to_cir(file, path):
    with open(path + file,'r') as file:
        QASM_str = file.read()
        cir = QuantumCircuit.from_qasm_str(QASM_str)
        cir.remove_final_measurements()
        cir = RemoveBarriers()(cir)
        # cir = transpile(cir, basis_gates=['h', 'x', 'p', 'cp', 'cswap', 'cx', 'swap'])
        cir = transpile(cir, basis_gates=['cz', 'u3'], optimization_level=2)
        print(cir.size())
        return cir

def generate_error(n, noisy_gate_num=0, random_pos = True):
    l = [i for i in range(80)]
    if random_pos == False:
        return [i for i in range(noisy_gate_num)]
    # return [0]
    return random.sample(l, noisy_gate_num)

def arr_to_tnvec(arr):
    vec = []
    for mat in arr:
        vec.append(tn.Node(np.array(mat)))
    return vec

def matrix_to_tensor(M):
    dim = int(math.log2(M.shape[0]))
    shape = (2,)*(2*dim)
    transpose_tuple = ()
    for i in range(dim):
        transpose_tuple += (2*i+1, 2*i)
    return np.transpose(np.reshape(M, shape), transpose_tuple)

def apply_gate(qubit_edges, gate, operating_qubits):
    op = tn.Node(gate)
    for i, bit in enumerate(operating_qubits):
        tn.connect(qubit_edges[bit], op[i])
        qubit_edges[bit] = op[i + len(operating_qubits)]

def apply_error2(qubits0, qubits1, errors):
    qubits = qubits0 + qubits1
    apply_gate(qubits, noise_gate, errors)
    qubits0[:] = qubits[:len(qubits0)]
    qubits1[:] = qubits[len(qubits0):]



# Apply an error circuit with double qubit, error_num=0 applies a non-error circuit, apply_inv 
# for circits with unknown output amd the output state is calculated by applying U to input state.
def error_cir_apply2(cir, qubits0, qubits1, error_num=0, error_pos=[], all_crz_fault = True, apply_inv = False):
    pos = 0
    cnt = 0
    cancel_out_gate = []

    for gate in cir.data:
        # Add gates to TN
        # print(gate[0].name)
        if(cnt == error_num and apply_inv):
            if(all_crz_fault and gate[0].name == 'cz'):
                for pre_gate in cancel_out_gate:
                    mat = pre_gate[0].to_matrix()
                    operating_qubits = [x.index for x in pre_gate[1]]
                    # print(mat)
                    # print(operating_qubits)
                    apply_gate(qubits0, mat, operating_qubits)
                    apply_gate(qubits1, mat.conjugate(), operating_qubits)
                cancel_out_gate = []
            else: 
                cancel_out_gate.append(gate)
                continue

        mat = gate[0].to_matrix()
        if(mat.shape[0]!=2):
            mat = matrix_to_tensor(mat)
        operating_qubits = [x.index for x in gate[1]]
        apply_gate(qubits0, mat, operating_qubits)
        apply_gate(qubits1, mat.conjugate(), operating_qubits)
        # CZ unitary fault
        if (all_crz_fault and gate[0].name == 'cz'):
            apply_gate(qubits0, fault_crz, operating_qubits)
            apply_gate(qubits1, fault_crz.conjugate(), operating_qubits)
        # Decoherence noise
        if(cnt < error_num and pos == error_pos[cnt]):
            error_qubit = random.sample(operating_qubits, 1)[0]
            errors = [error_qubit, error_qubit + len(qubits0)]
            apply_error2(qubits0, qubits1, errors)
            cnt = cnt + 1   
        pos = pos + 1
        # print(pos)

    if (apply_inv == False): return 

    cir_inv = cir.inverse()
    inv_pos = 0
    cancel_count = len(cancel_out_gate)
    for gate in cir_inv.data:
        if inv_pos < cancel_count:
            inv_pos += 1
            continue
        mat = gate[0].to_matrix()
        if(mat.shape[0]!=2):
            mat = matrix_to_tensor(mat)
        operating_qubits = [x.index for x in gate[1]]
        apply_gate(qubits0, mat, operating_qubits)
        apply_gate(qubits1, mat.conjugate(), operating_qubits)
    return