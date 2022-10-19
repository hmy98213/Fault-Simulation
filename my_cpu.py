from ast import Try
import math
import os
import random
import time
import numpy as np
from random import randrange
import tensornetwork as tn
from qiskit_fidelity import *
import gc
import signal
from contextlib import contextmanager
from qiskit.transpiler.passes import RemoveBarriers
class TimeoutException(Exception): pass

def Rz(theta):
    mat = np.zeros((2, 2), dtype=complex)
    mat[0][0] = np.exp(-1.0j * theta /2)
    mat[1][1] = np.exp(1.0j * theta /2)
    return np.array(mat)

noise_gate = np.zeros((2, 2, 2, 2), dtype=complex)
noise_gate[0][0][0][0] = 1
noise_gate[0][1][0][1] = noise_gate[1][0][1][0] = 0.98224288
noise_gate[1][1][1][1] = 0.99750312
noise_gate[1][0][0][1] = 0.00249688
noise_gate = np.array(noise_gate)

fault_rz = Rz(0.06)

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def file_to_cir(file, path):
    with open(path + file,'r') as file:
        QASM_str = file.read()
        cir = QuantumCircuit.from_qasm_str(QASM_str)
        cir.remove_final_measurements()
        cir = RemoveBarriers()(cir)
        # cir = transpile(cir, basis_gates=['h', 'x', 'p', 'cp', 'cswap', 'cx', 'swap'])
        # cir = transpile(cir, basis_gates=['cz', 'u3'], optimization_level=3)
        print(cir.size())
        return cir

def generate_error(n, noisy_gate_num=0, random_pos = True):
    l = [i for i in range(n)]
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


def apply_error2(qubits0, qubits1, errors, type):
    if type == 'noise':
        qubits = qubits0 + qubits1
        apply_gate(qubits, noise_gate, errors)
        qubits0[:] = qubits[:len(qubits0)]
        qubits1[:] = qubits[len(qubits0):]
    else:
        apply_gate(qubits0, fault_rz, errors[0])
        apply_gate(qubits1, fault_rz, errors[1])


# Apply an error circuit with double qubit, error_num=0 applies a non-error circuit, apply_inv 
# for circits with unknown output amd the output state is calculated by applying U to input state.
def error_cir_apply2(cir, qubits0, qubits1, error_num=0, error_pos=[], apply_inv = False):
    pos = 0
    cnt = 0
    for gate in cir.data:
        mat = gate[0].to_matrix()
        if(mat.shape[0]!=2):
            mat = matrix_to_tensor(mat)
        operating_qubits = [x.index for x in gate[1]]
        apply_gate(qubits0, mat, operating_qubits)
        apply_gate(qubits1, mat.conjugate(), operating_qubits)
        
        if(cnt < error_num and pos == error_pos[cnt]):
            error_qubit = random.sample(operating_qubits, 1)[0]
            errors = [error_qubit, error_qubit + len(qubits0)]
            apply_error2(qubits0, qubits1, errors, type = 'unitary')
            cnt = cnt + 1        
            if(cnt == error_num and apply_inv):
                cir_inv = cir.inverse()
                gate_num = cir.size()
                inv_pos = 0
                for gate in cir_inv.data:
                    if inv_pos < gate_num - error_pos[-1] - 1:
                        inv_pos += 1
                        continue
                    mat = gate[0].to_matrix()
                    if(mat.shape[0]!=2):
                        mat = matrix_to_tensor(mat)
                    operating_qubits = [x.index for x in gate[1]]
                    apply_gate(qubits0, mat, operating_qubits)
                    apply_gate(qubits1, mat.conjugate(), operating_qubits)
                return

        pos = pos + 1

#calculate fidelity with only one input: \ket{00...0}
def cal_fidelity_in(cir, ps1, error_num, error_pos):
    all_nodes = []
    with tn.NodeCollection(all_nodes):
        right_vec0 = arr_to_tnvec(ps1)
        right_vec1 = arr_to_tnvec(ps1)
        left_vec0 = arr_to_tnvec(ps1)
        left_vec1 = arr_to_tnvec(ps1)
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
        error_cir_apply2(cir, qubits0, qubits1, error_num, error_pos, apply_inv=True)
        # error_cir_apply2(cir.inverse(), qubits0, qubits1)

        for i in range(cir.num_qubits):
            tn.connect(start_wires0[i], left_vec0[i][0])
            tn.connect(qubits0[i], right_vec0[i][0])
            tn.connect(start_wires1[i], left_vec1[i][0])
            tn.connect(qubits1[i], right_vec1[i][0])
        time_now = datetime.datetime.now()
        print(time_now.strftime('%m.%d-%H:%M:%S'))

    return tn.contractors.auto(all_nodes).tensor



def file_test(path, file_name, output, error_num = 0):
    f = open(output, 'a')
    f.write("\n")
    time_now = datetime.datetime.now()
    print(time_now.strftime('%m.%d-%H:%M:%S'))

    cir = file_to_cir(file_name, path)
    nqubits = cir.num_qubits
    gate_num = cir.size()
    dep = cir.depth()

    error_pos = generate_error(gate_num, error_num ,random_pos=False)
    error_pos.sort()
    print(error_pos)
    # error_pos = [0, 1]

    ps1 = [np.array([1.0, 0], dtype=complex) for i in range(nqubits)]
    ps2 = [np.array([1.0, 0], dtype=complex) for i in range(nqubits)]

    if file_name.startswith('qft'):
        ps2 = [np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex) for i in range(nqubits)]
    elif file_name.startswith('bv'):
        ps2 = [np.array([0, 1.0], dtype = complex) for i in range(nqubits)]
        ps2[-1] = np.array([1/ np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)

    # file_name = file_name.replace('.qasm', '')
    print('circuit:', file_name)
    f.write(file_name+"\t")

    print('qubits:', nqubits)
    f.write(str(nqubits)+"\t")

    print('gates number:', gate_num)
    f.write(str(gate_num)+"\t")

    print('depth:', dep)
    f.write(str(dep)+"\t")

    print('noisy_num:', error_num)
    f.write(str(error_num)+"\t")

    try:
        t_start = time.time()
        result = np.real(cal_fidelity_in(cir, ps1, error_num, error_pos))
        result = np.sqrt(result)
        run_time = time.time() - t_start
        print("alg2 run time: ", run_time)
        f.write(str(run_time)+"\t")
        print(np.sqrt(result))
        f.write(str(result)+"\t")
    except TimeoutException as e:
        f.write(str(e)+"\n")
    except Exception as e:
        raise
        f.write(str(e)+"\n")
    f.close()


def folder_test(path, output_file, error_num = 0):
    files = os.listdir(path)
    for f in files:
        if f.startswith('gf'):
            continue
        try:
            file_test(path, f, output_file, error_num)
        except:
            pass

def noise_number_test(path, file_name, output_file):
    for noise_number in range(0, 16, 2):
        try:
            file_test(path, file_name, output_file, noise_number)
        except:
            raise
            pass

if __name__ == '__main__':
    error_num = 2
    # tn.set_default_backend("pytorch")
    # noise_number_test("Benchmarks/QAOA2/", "qaoa_100.qasm", "TN_qaoa_result.txt")
    # noise_number_test("Benchmarks/inst_TN/", "inst_6x6_20_0.qasm", "TN_inst_result.txt")
    # file_test("Benchmarks/inst_TN/", "inst_6x6_20_0.qasm", "TN_inst_result.txt", error_num)
    file_test("Benchmarks/BV/", "bv_n10.qasm", "TN_inst_result.txt", error_num)

    # folder_test('Benchmarks/QAOA/', "TN_result_qaoa.txt", error_num)
    # folder_test('Benchmarks/HFVQE/', "TN_result_hfvqe.txt", error_num)
    # folder_test('Benchmarks/inst_TN/', "TN_result_inst.txt", error_num)




