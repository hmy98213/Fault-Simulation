from qiskit import *
from mqt import ddsim

from cir_input.circuit_DG import CreateDGfromQASMfile

file_name = 'inst_7x7_10_0.qasm'
path = 'Benchmarks/inst/'

circ, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)

# print(circ.draw(fold=-1))

backend = ddsim.DDSIMProvider().get_backend('statevector_simulator')

job = execute(circ, backend)
# result = job.result()
# statevector = result.get_statevector(circ)
# print(statevector)
