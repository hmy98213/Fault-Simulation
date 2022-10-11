# Import tools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit_sliqsim_provider import SliQSimProvider

# Initiate SliQSim Provider
provider = SliQSimProvider()

# Construct a 2-qubit bell-state circuit
qr = QuantumRegister(2)
cr = ClassicalRegister(2)
qc = QuantumCircuit(qr, cr)

qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.measure(qr, cr)

# Get the backend of weak simulation
backend = provider.get_backend('weak_simulator')

# Execute simulation
job = execute(qc, backend=backend, shots=1024)

# Obtain and print the results
result = job.result()
print(result.get_counts(qc))