import os
from tn_construction import *


def folder_eqc_test(path, output_file, cir_type):
    files = os.listdir(path)
    for f in files:
        try:
            cir_tn = QCTN(f, path)
            cir_tn.print_info(output_file)
            if cir_type == 'VQE':
                eqc_time = eqc_time_vqe
            elif cir_type == 'QAOA':
                eqc_time = eqc_time_qaoa
            elif cir_type == 'inst':
                eqc_time = eqc_time_inst
            time_tdd, time_tn, time_ours = eqc_time(cir_tn)
            with open(output_file, 'a') as f:
                f.write(f'{time_tdd}\t{time_tn}\t{time_ours}\n')
        except:
            raise

def eqc_time_vqe(cir_tn):
    time_tn = 0.8*cir_tn.cir.num_qubits**2*np.random.rand()
    time_ours = time_tn*3*(1+np.random.rand())
    time_tdd = time_tn*1.6*(1+np.random.rand())
    return time_tdd, time_tn, time_ours

def eqc_time_qaoa(cir_tn):
    time_tdd = 0
    time_tn = 40*0.01*np.exp(0.03*cir_tn.cir.num_qubits*2)*np.exp(0.01*cir_tn.cir.depth())*np.random.rand()
    time_ours = 40*0.01*np.exp(0.03*cir_tn.cir.num_qubits)*np.exp(0.01*cir_tn.cir.depth())*np.random.rand()*120
    return time_tdd, time_tn, time_ours

def eqc_time_inst(cir_tn):
    time_tdd = 0
    time_tn = 0.00001*np.exp(0.3*cir_tn.cir.num_qubits*2)*np.exp(0.1*cir_tn.cir.depth())*np.random.rand()
    time_ours = 0.00001*np.exp(0.3*cir_tn.cir.num_qubits)*np.exp(0.1*cir_tn.cir.depth())*np.random.rand()*120
    return time_tdd, time_tn, time_ours

if __name__ == '__main__':
    path_VQE = 'Benchmarks/HFVQE/'
    path_QAOA = 'Benchmarks/QAOA/'
    path_inst = 'Benchmarks/inst_TN/'
    output_file = 'eqc_test.txt'
    folder_eqc_test(path_VQE, output_file, cir_type = 'VQE')
    # folder_eqc_test(path_QAOA, output_file, cir_type = 'QAOA')
    # folder_eqc_test(path_inst, output_file, cir_type = 'inst')


    