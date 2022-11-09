from contextlib import contextmanager
import datetime
import gc
import os
import signal
import time
from qiskit import *
from mqt import ddsim
from my_cpu import TimeoutException, file_to_cir
import numpy as np

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

def file_test(path, file_name, output, error_num = 0):
    f = open(output, 'a')
    f.write("\n")
    time_now = datetime.datetime.now()
    print(time_now.strftime('%m.%d-%H:%M:%S'))

    circ = file_to_cir(file_name, path)
    nqubits = circ.num_qubits
    gate_num = circ.size()
    dep = circ.depth()
    print('circuit:', file_name)
    f.write(file_name+"\t")

    print('qubits:', nqubits)
    f.write(str(nqubits)+"\t")

    print('gates number:', gate_num)
    f.write(str(gate_num)+"\t")

    print('depth:', dep)
    f.write(str(dep)+"\t")

    try:
        with time_limit(3600):
            backend = ddsim.DDSIMProvider().get_backend('qasm_simulator')
            t_start = time.time()
            job = execute(circ, backend, shots=1)
            # res = job.result()
            # print(res)
            counts = job.result().get_counts(circ)
            print(counts)   
            run_time = time.time() - t_start
            print("ddsim run time: ", run_time)
            f.write(str(run_time)+"\t")
            # print(np.sqrt(res))
            # f.write(str(res)+"\t")
    except TimeoutException as e:
        f.write(str(e)+"\n")
    except Exception as e:
        raise
        f.write(str(e)+"\n")
    f.close()
    
def folder_test(path, output_file, error_num = 0):
    files = os.listdir(path)
    for f in files:
        try:
            file_test(path, f, output_file, error_num)
        except:
            raise

if __name__ == "__main__":
    # file_test('Benchmarks/inst_TN/', "inst_4x5_20_0.qasm", "DDSIM_result.txt")
    folder_test('Benchmarks/QAOA2/', "DDSIM_result.txt")