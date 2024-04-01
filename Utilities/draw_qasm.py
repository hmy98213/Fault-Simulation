from qiskit import QuantumCircuit

def draw_qaoa():
    with open("pic.qasm", 'r') as f:
        qasm_str = f.read()
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    return qc.draw('latex')

if __name__ == "__main__":
    draw_qaoa()