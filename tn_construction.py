import gc
from qiskit import QuantumCircuit
import math
import random
import time
import numpy as np
import tensornetwork as tn
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
class TimeoutException(Exception): pass
from error_gen import *
from config import *
import warnings
from verification_by_qiskit import qiskit_simulate
warnings.filterwarnings("ignore")

# Class for the tensor network of a noisy quantum circuit
#   cir: quantum circuit
#   error_num: Number of noises
#   error_pos: Positions of noises
#   ps1: input state, must be a tensor product state, represented by a list of numpy arrays, 
# e.g. ps1 = [np.array([1.0, 0], dtype=complex), np.array([1.0, 0], dtype=complex)] represents zero state
#   ps2: measurement defined by a state, also must be a tensor product state
#   ps1 and ps2 are all default to be zero state.
class QCTN:
    cir = None # Quantum circuit
    all_nodes = [] # Tensor nodes
    noise_gate = gen_noise_gate(T_ONE_TIME, T_TWO_TIME, DELTA_T) # Noise gate
    mat_op = mat_rep_svd(noise_gate) # SVD of noise gate
    error_num = 0 # Number of errors
    error_pos = [] # Positions of errors
    ps1 = [] # Input state
    ps2 = [] # State defining the output

    def __init__(self, file, path):
        self.file_name = file
        self.cir = file_to_cir(file, path)
        ket_zero = [np.array([1.0, 0], dtype=complex) for _ in range(self.cir.num_qubits)]
        self.ps1 = ket_zero.copy(); self.ps2 = ket_zero.copy()
        self.all_nodes = []

    # Set input and output states
    def set_io(self, ps1, ps2):
        self.ps1 = ps1.copy()
        self.ps2 = ps2.copy()

    # Set decoherence time
    def set_decoherence_time(self, T1, T2):
        self.noise_gate = gen_noise_gate(T1, T2, DELTA_T)
        self.mat_op = mat_rep_svd(self.noise_gate)

    # Set error number and positions
    def set_error(self, error_num, random_pos = True):
        self.error_num = error_num
        l = [i for i in range(self.cir.size())]
        if random_pos == False:
            self.error_pos = NOISE_POS
        else:
            self.error_pos = random.sample(l, error_num)
        self.error_pos.sort()

    # Related to the paper: "Approximation Algorithm for Noisy Quantum Circuit Simulation"
    def gen_idx(self, N, n):
        if(n==0): return [[0]*N]
        res = []
        for e in self.gen_idx(N, n-1):
            first_nonzero = True if n==1 else False
            for i in range(N):
                if (first_nonzero == False and e[i]==0): continue
                if (first_nonzero == False and e[i]!=0): first_nonzero = True
                if (e[i]==0): 
                    # Karus operators other than the one almost equal to identity
                    for k in range(1, 4):
                        tmp = e.copy()
                        tmp[i] = k
                        res.append(tmp)
        return res

    def idx_to_op(self, id):
        res_u = []
        res_v = []
        res_d = []
        for i in id:
            res_u.append(self.mat_op[0][i])
            res_v.append(self.mat_op[2][i])
            res_d.append(self.mat_op[1][i])
        return [res_u, res_d, res_v]

    def print_info(self, output_file):
        with open(output_file, 'a') as f:
            nqubits = self.cir.num_qubits
            gate_num = self.cir.size()
            dep = self.cir.depth()
            file_name = self.file_name.replace('.qasm', '')
            print('circuit:', file_name)
            f.write(file_name+"\t")
            print('qubits:', nqubits)
            f.write(str(nqubits)+"\t")
            print('gates number:', gate_num)
            f.write(str(gate_num)+"\t")
            print('depth:', dep)
            f.write(str(dep)+"\t")
            print('noisy_num:', self.error_num)
            print(self.error_pos)
            f.write(str(self.error_num)+"\t")

    # Apply a noiseless circuit to a tensor network
    def cir_apply(self, qubits):
        for gate in self.cir.data:
            mat = np.matrix(gate[0].to_matrix()).H
            mat = np.array(mat)
            if(mat.shape[0]!=2): mat = matrix_to_tensor(mat)
            operating_qubits = [x.index for x in gate[1]]
            apply_gate(qubits, mat, operating_qubits)

    # Apply an noisy quantum circuit, with noise substituted by a matrix obtained from the SVD (approximate algorithm)
    def error_cir_apply_enumurate(self, qubits, op_list):
        cnt = 0
        pos = 0
        for gate in self.cir.data:
            mat = np.matrix(gate[0].to_matrix()).H
            mat = np.array(mat)
            if(mat.shape[0]!=2): mat = matrix_to_tensor(mat)
            operating_qubits = [x.index for x in gate[1]]
            apply_gate(qubits, mat, operating_qubits)
            # Decoherence noise
            if(cnt < self.error_num and pos == self.error_pos[cnt]):
                error_qubit = operating_qubits[0]
                errors = [error_qubit]
                error_gate = np.array(np.matrix(op_list[cnt]).H)
                apply_gate(qubits, error_gate, errors)
                cnt += 1
            pos += 1

    # Apply an noisy quantum circuit and its reverse, with noise substituted by a matrix obtained from the SVD,
    # and gates in the end of the circuit are cancelled (approximate algorithm)
    def error_cir_inv_apply_enumurate(self, qubits, op_list):
        pos = 0
        cnt = 0
        for gate in self.cir.data:
            if(pos == self.error_pos[-1] + 1): break
            mat = np.matrix(gate[0].to_matrix()).H
            mat = np.array(mat)
            if(mat.shape[0]!=2): mat = matrix_to_tensor(mat)
            operating_qubits = [x.index for x in gate[1]]
            apply_gate(qubits, mat, operating_qubits)
            # Decoherence noise
            if(cnt < self.error_num and pos == self.error_pos[cnt]):
                error_qubit = operating_qubits[0]
                errors = [error_qubit]
                error_gate = np.array(np.matrix(op_list[cnt]).H)
                apply_gate(qubits, error_gate, errors)
                cnt += 1
            pos += 1
        inv_pos = 0
        cir_inv = self.cir.inverse()
        cancel_count = len(self.cir.data) - self.error_pos[-1] - 1
        for gate in cir_inv.data:
            if inv_pos < cancel_count:
                inv_pos += 1
                continue
            mat = np.matrix(gate[0].to_matrix()).H
            mat = np.array(mat)
            if(mat.shape[0]!=2):
                mat = matrix_to_tensor(mat)
            operating_qubits = [x.index for x in gate[1]]
            apply_gate(qubits, mat, operating_qubits)

    # Apply an error circuit with double qubits, with noise substituted by the matrix representation of it (accurate algorithm)
    def error_cir_apply(self, qubits0, qubits1):
        pos = 0
        cnt = 0
        for gate in self.cir.data:
            mat = np.matrix(gate[0].to_matrix()).H
            mat = np.array(mat)
            if(mat.shape[0]!=2):
                mat = matrix_to_tensor(mat)
            operating_qubits = [x.index for x in gate[1]]
            apply_gate(qubits0, mat, operating_qubits)
            apply_gate(qubits1, mat.conjugate(), operating_qubits)       
            # Decoherence noise
            if(cnt < self.error_num and pos == self.error_pos[cnt]):
                error_qubit = operating_qubits[0]
                errors = [error_qubit, error_qubit + len(qubits0)]
                noise_gateH = np.reshape(np.array(np.matrix(np.reshape(self.noise_gate, (4, 4))).H), (2, 2, 2, 2))
                apply_error2(qubits0, qubits1, noise_gateH, errors)
                cnt = cnt + 1   
            pos = pos + 1

    def error_cir_inv_apply(self, qubits0, qubits1):
        pos = 0
        cnt = 0
        cancel_out_gate = []

        for gate in self.cir.data:
            # Add gates to TN
            if(pos == self.error_pos[-1] + 1): break
            mat = np.matrix(gate[0].to_matrix()).H
            mat = np.array(mat)
            if(mat.shape[0]!=2): mat = matrix_to_tensor(mat)
            operating_qubits = [x.index for x in gate[1]]
            apply_gate(qubits0, mat, operating_qubits)
            apply_gate(qubits1, mat.conjugate(), operating_qubits)
            # Decoherence noise
            if(cnt < self.error_num and pos == self.error_pos[cnt]):
                error_qubit = operating_qubits[0]
                errors = [error_qubit, error_qubit + len(qubits0)]
                apply_error2(qubits0, qubits1, self.noise_gate ,errors)
                cnt = cnt + 1   
            pos = pos + 1

        cir_inv = self.cir.inverse()
        inv_pos = 0
        cancel_count = len(cancel_out_gate)
        for gate in cir_inv.data:
            if inv_pos < cancel_count:
                inv_pos += 1
                continue
            mat = np.matrix(gate[0].to_matrix()).H
            mat = np.array(mat)
            if(mat.shape[0]!=2):
                mat = matrix_to_tensor(mat)
            operating_qubits = [x.index for x in gate[1]]
            apply_gate(qubits0, mat, operating_qubits)
            apply_gate(qubits1, mat.conjugate(), operating_qubits)

    def construct_tn_no_error(self, ps1, ps2):
       with tn.NodeCollection(self.all_nodes):
            left_vec= arr_to_tnvec(ps1)
            right_vec = arr_to_tnvec(ps2)
            start_gates = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]

            qubits = [node[1] for node in start_gates]
            start_wires = [node[0] for node in start_gates]
            self.cir_apply(qubits)

            for i in range(self.cir.num_qubits):
                tn.connect(start_wires[i], left_vec[i][0])
                tn.connect(qubits[i], right_vec[i][0])

    def construct_enumurate_tn(self, ps1, ps2, op_list):
       with tn.NodeCollection(self.all_nodes):
            left_vec= arr_to_tnvec(ps1)
            right_vec = arr_to_tnvec(ps2)
            start_gates = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]

            qubits = [node[1] for node in start_gates]
            start_wires = [node[0] for node in start_gates]
            self.error_cir_apply_enumurate(qubits, op_list)

            for i in range(self.cir.num_qubits):
                tn.connect(start_wires[i], left_vec[i][0])
                tn.connect(qubits[i], right_vec[i][0])

    def construct_enumurate_tn_inv(self, ps1, op_list):
        with tn.NodeCollection(self.all_nodes):
            left_vec= arr_to_tnvec(ps1)
            right_vec = arr_to_tnvec(ps1)
            start_gates = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]

            qubits = [node[1] for node in start_gates]
            start_wires = [node[0] for node in start_gates]
            self.error_cir_inv_apply_enumurate(qubits, op_list)

            for i in range(self.cir.num_qubits):
                tn.connect(start_wires[i], left_vec[i][0])
                tn.connect(qubits[i], right_vec[i][0])

    def construct_tn(self, ps1, ps2):
        with tn.NodeCollection(self.all_nodes):
            left_vec0 = arr_to_tnvec(ps1)
            left_vec1 = arr_to_tnvec(ps1)
            right_vec0 = arr_to_tnvec(ps2)
            right_vec1 = arr_to_tnvec(ps2)
            start_gates0 = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]
            start_gates1 = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]
            qubits0 = [node[1] for node in start_gates0]
            qubits1 = [node[1] for node in start_gates1]
            start_wires0 = [node[0] for node in start_gates0]
            start_wires1 = [node[0] for node in start_gates1]
            self.error_cir_apply(qubits0, qubits1)
            for i in range(self.cir.num_qubits):
                tn.connect(start_wires0[i], left_vec0[i][0])
                tn.connect(qubits0[i], right_vec0[i][0])
                tn.connect(start_wires1[i], left_vec1[i][0])
                tn.connect(qubits1[i], right_vec1[i][0])

    def construct_tn_inv(self, ps1):
        with tn.NodeCollection(self.all_nodes):
            right_vec0 = arr_to_tnvec(ps1)
            right_vec1 = arr_to_tnvec(ps1)
            left_vec0 = arr_to_tnvec(ps1)
            left_vec1 = arr_to_tnvec(ps1)
            start_gates0 = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]
            start_gates1 = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]
            qubits0 = [node[1] for node in start_gates0]
            qubits1 = [node[1] for node in start_gates1]
            start_wires0 = [node[0] for node in start_gates0]
            start_wires1 = [node[0] for node in start_gates1]
            self.error_cir_inv_apply(qubits0, qubits1)

            for i in range(self.cir.num_qubits):
                tn.connect(start_wires0[i], left_vec0[i][0])
                tn.connect(qubits0[i], right_vec0[i][0])
                tn.connect(start_wires1[i], left_vec1[i][0])
                tn.connect(qubits1[i], right_vec1[i][0])

    # Circuit for equivalence checking (accurate algorithm)
    def construct_tn_eqc(self, ps1):
        with tn.NodeCollection(self.all_nodes):
            right_vec0 = arr_to_tnvec(ps1)
            right_vec1 = arr_to_tnvec(ps1)
            left_vec0 = arr_to_tnvec(ps1)
            left_vec1 = arr_to_tnvec(ps1)
            start_gates0 = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]
            start_gates1 = [
                tn.Node(np.eye(2, dtype=complex)) for _ in range(self.cir.num_qubits)
            ]
            qubits0 = [node[1] for node in start_gates0]
            qubits1 = [node[1] for node in start_gates1]
            start_wires0 = [node[0] for node in start_gates0]
            start_wires1 = [node[0] for node in start_gates1]
            self.error_cir_inv_apply(qubits0, qubits1)

            for i in range(self.cir.num_qubits):
                tn.connect(start_wires0[i], left_vec0[i][0])
                tn.connect(qubits0[i], right_vec0[i][0])
                tn.connect(start_wires1[i], left_vec1[i][0])
                tn.connect(qubits1[i], right_vec1[i][0])

    def simu(self, output, ps1 = [], ps2= []):
        self.all_nodes = []
        if (len(ps1)>0 or len(ps2)>0):
            self.set_io(ps1, ps2) 
        try:
            self.construct_tn_no_error(self.ps1, self.ps2)
            t_start = time.time()
            # For test
            result = np.abs(tn.contractors.auto(self.all_nodes).tensor)**2
            with open(output, 'a') as f:
                run_time = time.time() - t_start
                print("run time: ", run_time)
                f.write(str(run_time)+"\t")
                print(result)
                f.write(str(result)+"\t")
                f.write("\n")
            return result
        except TimeoutException as e:
            with open(output, 'a') as f:
                f.write(str(e)+"\n")
        except Exception as e:
            # raise
            with open(output, 'a') as f:
                f.write(str(e)+"\n")

    def enumurate_io(self, output, error_num = 0, random_pos = True, ps1 = [], ps2= [], enum = 1):
        self.set_error(error_num, random_pos)
        self.print_info(output)
        if (len(ps1)>0 or len(ps2)>0):
            self.set_io(ps1, ps2) 
        try:
            t_start = time.time()
            res = 0
            for e in range(enum+1):
                for id in self.gen_idx(error_num, e):
                    gc.collect()
                    op_list = self.idx_to_op(id)
                    self.all_nodes = []
                    self.construct_enumurate_tn(self.ps1, self.ps2, op_list[0])
                    tmp0 = tn.contractors.auto(self.all_nodes).tensor
                    self.all_nodes = []
                    self.construct_enumurate_tn(self.ps1, self.ps2, op_list[2])
                    tmp1 = tn.contractors.auto(self.all_nodes).tensor.conjugate()
                    tmp = tmp0 * tmp1
                    for i in op_list[1]: tmp *= i
                    res += tmp
            # res = np.abs(res)
            # res = np.sqrt(res)
            with open(output, 'a') as f:
                run_time = time.time() - t_start
                print("run time: ", run_time)
                f.write(str(run_time)+"\t")
                print(res)
                f.write(str(res)+"\t")
                f.write("\n")
            return res
        except TimeoutException as e:
            with open(output, 'a') as f:
                f.write(str(e)+"\n")
        except Exception as e:
            raise
            # with open(output, 'a') as f:
            #     f.write(str(e)+"\n")

    def enumurate_fidility_test(self, output, error_num = 0, random_pos = True, ps1 = [], enum = 1):
        self.set_error(error_num, random_pos)
        self.print_info(output)
        if (len(ps1)>0):
            self.set_io(ps1, []) 
        try:
            t_start = time.time()
            res = 0
            for e in range(enum+1):
                for id in self.gen_idx(error_num, e):
                    gc.collect()
                    op_list = self.idx_to_op(id)
                    self.all_nodes = []
                    self.construct_enumurate_tn_inv(self.ps1, op_list[0])
                    tmp0 = tn.contractors.auto(self.all_nodes).tensor
                    self.all_nodes = []
                    for i in range(len(op_list[2])):
                        op_list[2][i] = op_list[2][i].conjugate()
                    self.construct_enumurate_tn_inv(self.ps1, op_list[2])
                    tmp1 = tn.contractors.auto(self.all_nodes).tensor
                    tmp = tmp0.item() * tmp1.item().conjugate()
                    for i in op_list[1]: tmp *= i
                    res += tmp
            res = np.abs(res)
            res = np.sqrt(res)
            with open(output, 'a') as f:
                run_time = time.time() - t_start
                print("run time: ", run_time)
                f.write(str(run_time)+"\t")
                print(res)
                f.write(str(res)+"\t")
                f.write("\n")
            return res
        except TimeoutException as e:
            with open(output, 'a') as f:
                f.write(str(e)+"\n")
        except Exception as e:
            with open(output, 'a') as f:
                f.write(str(e)+"\n")

    def io_test(self, output, error_num = 0, random_pos = True, ps1 = [], ps2 = []):
        self.set_error(error_num, random_pos)
        self.print_info(output)
        if (len(ps1)>0 or len(ps2)>0):
            self.set_io(ps1, ps2) 
        try:
            self.construct_tn(self.ps1, self.ps2)
            t_start = time.time()
            result = tn.contractors.auto(self.all_nodes).tensor
            with open(output, 'a') as f:
                run_time = time.time() - t_start
                print("run time: ", run_time)
                f.write(str(run_time)+"\t")
                print(result)
                f.write(str(result)+"\n")
            return result
        except TimeoutException as e:
            with open(output, 'a') as f:
                f.write(str(e)+"\n")
        except Exception as e:
            # raise
            with open(output, 'a') as f:
                f.write(str(e)+"\n")

    def fidelity_test(self, output, error_num = 0, random_pos = True, ps1 = []):
        self.set_error(error_num, random_pos)
        self.print_info(output)
        if (len(ps1)>0):
            self.set_io(ps1, [])
        try:
            if (error_num == 0):
                self.construct_tn_single_inv(self.ps1)
            else:
                self.construct_tn_inv(self.ps1)
            t_start = time.time()
            result = np.real(tn.contractors.greedy(self.all_nodes).tensor.item())
            result = np.sqrt(result) if error_num > 0 else result
            with open(output, 'a') as f:
                run_time = time.time() - t_start
                print("run time: ", run_time)
                f.write(str(run_time)+"\t")
                print(result)
                f.write(str(result)+"\t")
                f.write("\n")
            return result
        except TimeoutException as e:
            with open(output, 'a') as f:
                f.write(str(e)+"\n")
        except Exception as e:
            with open(output, 'a') as f:
                f.write(str(e)+"\n")

    def basis_test(self, output):
        f = open(output, 'a')
        f.write("\n")
        nqubits = self.cir.num_qubits
        for j in range(2**nqubits):
            f.write(str(j)+"\t")
            tmp_in = j
            tmp_ps1 = []
            for _ in range(nqubits):
                if tmp_in%2 == 0:
                    tmp_ps1.insert(0, np.array([1.0, 0], dtype=complex))
                else:
                    tmp_ps1.insert(0, np.array([0, 1.0], dtype=complex))
                tmp_in = tmp_in//2

            for i in range(2**nqubits):
                tmp = i
                tmp_ps2 = []
                for _ in range(nqubits):
                    if tmp%2 == 0:
                        tmp_ps2.insert(0, np.array([1.0, 0], dtype=complex))
                    else:
                        tmp_ps2.insert(0, np.array([0, 1.0], dtype=complex))
                    tmp = tmp//2
                try:
                    # Comment the print_info in io_test to get a compact output
                    self.all_nodes = []
                    result = self.io_test(output, ps1=tmp_ps1, ps2=tmp_ps2)**2
                    f.write("%.4f\t"%result)
                except TimeoutException as e:
                    f.write(str(e)+"\n")
                except Exception as e:
                    raise
                    # f.write(str(e)+"\n")
            f.write("\n")
        f.close()


def file_to_cir(file, path):
    with open(path + file,'r') as file:
        QASM_str = file.read()
        cir = QuantumCircuit.from_qasm_str(QASM_str)
        cir.remove_final_measurements()
        cir = RemoveBarriers()(cir)
        # cir = transpile(cir, basis_gates=['h', 'x', 'p', 'cp', 'cswap', 'cx', 'swap'])
        # cir = transpile(cir, basis_gates=['cz', 'u3'], optimization_level=3)
        # print(cir.size())
        return cir

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

def apply_error2(qubits0, qubits1, noise_gate, error_qubits):
    qubits = qubits0 + qubits1
    apply_gate(qubits, noise_gate, error_qubits)
    qubits0[:] = qubits[:len(qubits0)]
    qubits1[:] = qubits[len(qubits0):]


if __name__ == '__main__':
    # test_tn = QCTN("inst_4x4_10_0.qasm", "Benchmarks/inst_TN/")
    output = 'test_result.txt'
    circuit_file = 'simple.qasm'
    folder = './Benchmarks/Simple/'
    test_tn = QCTN(circuit_file, folder)
    # Noiseless Simulation
    test_tn.simu(output)
    qiskit_simulate(folder, circuit_file, error_num = 0, error_position=[]) # Verify the result with qiskit
    # Noise Simulation (Accurate), noise config in config.py
    test_tn = QCTN(circuit_file, folder)
    test_tn.io_test(output, error_num = NOISE_NUM, random_pos = False)
    qiskit_simulate(folder, circuit_file, error_num = NOISE_NUM, error_position=NOISE_POS) # Verify the result with qiskit
    # Noise Simulation (Approximate), noise config in config.py, enum is the level of enumutate
    test_tn = QCTN(circuit_file, folder)
    test_tn.enumurate_io(output, error_num = NOISE_NUM, random_pos = False, enum = 1)
