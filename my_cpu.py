from ast import Try
import datetime
import gc
import math
import os
import random
import time
import scipy
import numpy as np
import tensornetwork as tn
import signal
from contextlib import contextmanager
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.transpiler.passes import RemoveBarriers
from angle_gen import faulty_gate_2
class TimeoutException(Exception): pass
from error_gen import *

# noise_gate = np.zeros((2, 2, 2, 2), dtype=complex)
# noise_gate[0][0][0][0] = 1
# noise_gate[0][1][0][1] = noise_gate[1][0][1][0] = 0.98224288
# noise_gate[1][1][1][1] = 0.99750312
# noise_gate[1][0][0][1] = 0.00249688
# noise_gate = np.array(noise_gate)

noise_gate = gen_noise_gate(200, 30, 0.5)
# noise_gate = np.reshape(noise_gate, (2, 2, 2, 2))

theta = 0.1
t1 = 200
t2 = 30
fault_crz = cRz(theta)

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
        error_cir_apply2(cir, qubits0, qubits1, error_num, error_pos, all_crz_fault=True, apply_inv=True)
        # error_cir_apply2(cir.inverse(), qubits0, qubits1)

        for i in range(cir.num_qubits):
            tn.connect(start_wires0[i], left_vec0[i][0])
            tn.connect(qubits0[i], right_vec0[i][0])
            tn.connect(start_wires1[i], left_vec1[i][0])
            tn.connect(qubits1[i], right_vec1[i][0])
        time_now = datetime.datetime.now()
        print(time_now.strftime('%m.%d-%H:%M:%S'))

    return tn.contractors.auto(all_nodes).tensor

def cal_fidelity_io(cir, ps1, ps2, error_num, error_pos):
    all_nodes = []
    with tn.NodeCollection(all_nodes):
        right_vec0 = arr_to_tnvec(ps2)
        right_vec1 = arr_to_tnvec(ps2)
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
        error_cir_apply2(cir, qubits0, qubits1, error_num, error_pos, all_crz_fault=True, apply_inv=False)
        # error_cir_apply2(cir.inverse(), qubits0, qubits1)

        for i in range(cir.num_qubits):
            tn.connect(start_wires0[i], left_vec0[i][0])
            tn.connect(qubits0[i], right_vec0[i][0])
            tn.connect(start_wires1[i], left_vec1[i][0])
            tn.connect(qubits1[i], right_vec1[i][0])

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

    error_pos = generate_error(gate_num, error_num ,random_pos=True)
    error_pos.sort()
    print(error_pos)
    # error_pos = [0, 1]

    ps1 = [np.array([1.0, 0], dtype=complex) for i in range(nqubits)]
    ps2 = [np.array([1.0, 0], dtype=complex) for i in range(nqubits)]

    # file_name = file_name.replace('.qasm', '')
    f.write(str(theta) + "\t")
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

def file_basis_test(path, file_name, output, error_num = 0):
    f = open(output, 'a')
    f.write("\n")
    time_now = datetime.datetime.now()
    print(time_now.strftime('%m.%d-%H:%M:%S'))

    cir = file_to_cir(file_name, path)
    nqubits = cir.num_qubits
    gate_num = cir.size()

    error_pos = generate_error(gate_num, error_num ,random_pos=True)
    error_pos.sort()
    print(error_pos)
    # error_pos = [0, 1]

    ps1 = []

    for j in range(2**nqubits):
        f.write(str(j)+"\t")
        tmp_in = j
        ps1 = []
        for _ in range(nqubits):
            if tmp_in%2 == 0:
                ps1.insert(0, np.array([1.0, 0], dtype=complex))
            else:
                ps1.insert(0, np.array([0, 1.0], dtype=complex))
            tmp_in = tmp_in//2

        for i in range(2**nqubits):
            tmp = i
            ps2 = []
            for _ in range(nqubits):
                if tmp%2 == 0:
                    ps2.insert(0, np.array([1.0, 0], dtype=complex))
                else:
                    ps2.insert(0, np.array([0, 1.0], dtype=complex))
                tmp = tmp//2
            try:
                result = np.real(cal_fidelity_io(cir, ps1, ps2, error_num, error_pos))
                # np.set_printoptions(suppress=True, precision=8)
                # result = np.sqrt(result)
                # print("alg2 run time: ", run_time)
                # f.write(str(run_time)+"\t")
                # print("base state:", i)
                # f.write(str(run_time)+"\t")
                # print(result)
                f.write("%.4f\t"%result)
            except TimeoutException as e:
                f.write(str(e)+"\n")
            except Exception as e:
                raise
                f.write(str(e)+"\n")
        f.write("\n")
    f.close()


def folder_test(path, output_file, error_num = 0):
    files = os.listdir(path)
    for f in files:
        try:
            file_test(path, f, output_file, error_num)
        except:
            pass
        gc.collect()


def noise_number_test(path, file_name, output_file):
    for noise_number in range(0, 26, 2):
        try:
            file_test(path, file_name, output_file, noise_number)
        except:
            raise
            pass
        gc.collect()


def angle_test(path, filename, output_file):
    global theta, fault_crz
    lambda10 = 1.0866
    lambda20 = -0.0866
    tmax = 77.0
    g = math.sqrt(2)/40
    thetaf = math.pi/2
    thetai = math.atan(2*g/(0.31))

    def theta_noise(t, t0, lambda10, lambda20): #Fourier approximation of Slepian using 2 elements
        lambda1 = (thetaf-thetai)/t0*lambda10
        lambda2 = (thetaf-thetai)/t0*lambda20
        if( 0 <= t < t0):
            return thetai + ((lambda1+lambda2)*t - lambda1*t0/(2*math.pi)*math.sin(2*math.pi*t/t0)-lambda2*t0/(4*math.pi)*math.sin(4*math.pi*t/t0))
        elif (t0 <= t <= 2*t0):
            return (thetaf) - ((lambda1+lambda2)*(t-t0) - lambda1*t0/(2*math.pi)*math.sin(2*math.pi*(t-t0)/t0)-lambda2*t0/(4*math.pi)*math.sin(4*math.pi*(t-t0)/t0))   
    def faulty_gate_2(noise, tmax, index):
        def theta_noise1_value(x):
            return math.tan(theta_noise(x, tmax/2, lambda10 * (1 + noise), lambda20)/2)
        
        def theta_noise2_value(x):
            return math.tan(theta_noise(x, tmax/2, lambda10 + noise, lambda20 )/2)
        def theta_value(x):
            return math.tan(theta_noise(x, tmax/2, lambda10, lambda20)/2)
        
        sg2 = 2 * math.pi * g
        varphi1 = scipy.integrate.quad(theta_noise1_value, 0, tmax)[0]
        varphi2 = scipy.integrate.quad(theta_noise2_value, 0, tmax)[0]
        varphi3 = scipy.integrate.quad(theta_value, tmax * abs(noise/2), tmax * (1 - abs(noise)/2))[0]
        varphi = scipy.integrate.quad(theta_value,0,tmax)[0]
        if(index == 1): 
            return sg2 * (varphi1 - varphi)
        elif(index == 2):
            return sg2 * (varphi2 - varphi)
        elif(index == 3):
            return sg2 * (varphi3 - varphi)

    num_samples = 9
    X = np.linspace(-0.2, 0.2, num_samples)
    Y1 = np.zeros(num_samples)
    Y2 = np.zeros(num_samples)
    Y3 = np.zeros(num_samples)
    for index, noise in enumerate(X):
        tmax = 77.0
        Y1[index] = faulty_gate_2(noise, tmax,1)
        Y2[index] = faulty_gate_2(noise, tmax,2)
        Y3[index] = faulty_gate_2(noise, tmax,3)
    for Y in [Y1, Y2, Y3]:
        for index, noise in enumerate(X):
            theta = Y[index]
            fault_crz = cRz(theta)
            with open(output_file, 'a') as f:
                f.write("%f\t%f\t"%(X[index], theta))
            file_basis_test(path, filename, output_file)


if __name__ == '__main__':
    error_num = 0
    tn.set_default_backend("pytorch")
    # print(gen_noise_gate(200, 30, 0.5))
    # x = np.linspace(0, 0.1, 50)
    # for theta in x:
    #     fault_crz = cRz(theta)
    #     # file_test("Benchmarks/inst_TN/", "inst_6x6_10_0.qasm", "TN_inst_result.txt", error_num)
    #     file_test("Benchmarks/QAOA2/", "qaoa_64.qasm", "TN_qaoa_result.txt", error_num)

    # print(gen_noise_gate(200, 30, 0.5))
    # x = np.linspace(25, 35, 50)
    # for t in x:
    #     t2 = t
    #     noise_gate = gen_noise_gate(200, t2, 0.5)
    #     file_test("Benchmarks/inst_TN/", "inst_6x6_20_0.qasm", "TN_inst_result.txt", error_num)
    #     file_test("Benchmarks/QAOA2/", "qaoa_100.qasm", "TN_qaoa_result.txt", error_num)


    # file_test("Benchmarks/QAOA2/", "qaoa_100.qasm", "TN_qaoa_result.txt", error_num)

    # noise_number_test("Benchmarks/QAOA2/", "qaoa_81.qasm", "TN_qaoa_result.txt")
    # noise_number_test("Benchmarks/inst_TN/", "inst_6x6_20_0.qasm", "TN_inst_result.txt")
    # file_test("Benchmarks/inst_TN/", "inst_6x6_40_0.qasm", "TN_inst_result.txt", error_num)
    
    
    # file_basis_test("Benchmarks/Simple/", "4-qubit-full-adder.qasm", "result.txt", error_num)

    # folder_test('Benchmarks/QAOA/', "TN_result_qaoa.txt", error_num)
    # folder_test('Benchmarks/HFVQE/', "TN_result_hfvqe.txt", error_num)
    # folder_test('Benchmarks/inst_TN/', "TN_result_inst.txt", error_num)

    angle_test("Benchmarks/Simple/", "4qbit-random-circ.qasm", "angle_result_rand.txt")
    angle_test("Benchmarks/Simple/", "4-qubit-full-adder.qasm", "angle_result_adder.txt")



