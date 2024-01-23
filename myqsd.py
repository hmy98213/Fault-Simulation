from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
import numpy as np

# Define your 8x8 unitary matrix
M = np.zeros((8, 8), dtype=complex)
for i in range(8):
    M[i, (i+1) % 8] = 1.0

# Create a quantum circuit from the unitary matrix
qc = QuantumCircuit(3)  # 3 qubits
qc.unitary(Operator(M), [0, 1, 2], label='U')

# Decompose the circuit into a standard gate set
decomposed_circuit = transpile(qc, basis_gates=['cx', 'rz', 'ry', 'u3'], optimization_level=3)

# Print the decomposed circuit
print(decomposed_circuit.qasm())
