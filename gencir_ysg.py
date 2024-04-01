from qiskit import QuantumCircuit

def gen_test_cir(nqbit, level):
    """Generate a test circuit with nqbit qubits"""
    qc = QuantumCircuit(nqbit)
    for i in range(nqbit):
        qc.h(i)
    for _ in range(level):
        for i in range(nqbit):
            qc.t(i)
        for i in range(nqbit-1):
            qc.cx(i, i+1)
    return qc

if __name__ == '__main__':
    qc1 = gen_test_cir(10, 100)
    qc2 = gen_test_cir(20, 5)
    qc3 = gen_test_cir(50, 5)
    with open('ysg10.qasm', 'w') as f:
        f.write(qc1.qasm())
    with open('ysg20.qasm', 'w') as f:
        f.write(qc2.qasm())
    with open('ysg50.qasm', 'w') as f:
        f.write(qc3.qasm())