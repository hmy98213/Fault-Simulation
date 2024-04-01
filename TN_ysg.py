import torch
from tn_construction import *
from gencir_ysg import *
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info.operators.channel import SuperOp
from qiskit_aer.noise import depolarizing_error
from qiskit.quantum_info import Statevector
import qiskit.quantum_info as qi


if __name__ =='__main__':
    torch.set_printoptions(precision=10)
    tn.set_default_backend("pytorch")
    output_file = 'ysg.txt'
    # # Use qiskit to verify the result
    # cir = file_to_cir('ysg20.qasm', '')
    # sv = Statevector.from_label('0'*20)
    # sv = sv.evolve(cir)
    # print(sv)
    # cir_tn = QCTN('ysg10.qasm', '')
    # cir_tn.simu(output_file)
    # cir_tn = QCTN('ysg20.qasm', '')
    # cir_tn.simu(output_file)
    cir_tn = QCTN('ysg50.qasm', '')
    cir_tn.simu(output_file)
    print('Finish!')